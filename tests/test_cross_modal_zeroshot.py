"""Run with: python -m unittest discover -s tests -p 'test_cross_modal_zeroshot.py'."""

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from sklearn.metrics import mean_squared_error, r2_score

from BCRFoundation.pretrain import BCRVocab, FoundationModel, MaskedPretrainingModel
from BCRFoundation.finetune.cross_modal_zeroshot import (
    load_zeroshot_regressor, predict_regressor,
)


class ZeroShotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "checkpoint.pt"
        cls.vocab = BCRVocab()
        cls.vocab.build(["ACDEFGHIK"])
        foundation = FoundationModel(3, len(cls.vocab.vocab), max_length=8)
        # Nontrivial learned normalization catches accidental bypasses.
        with torch.no_grad():
            foundation.norm_bcr_l.weight.fill_(1.7)
            foundation.norm_bcr_l.bias.fill_(0.3)
            foundation.norm_bcr_h.weight.fill_(0.6)
            foundation.norm_bcr_h.bias.fill_(-0.4)
        cls.pretrained = MaskedPretrainingModel(foundation, 3, 8, len(cls.vocab.vocab))
        cls.pretrained.eval()
        cls.checkpoint = {
            "model_state_dict": foundation.state_dict(),
            "bcr2gene_decoder_state_dict": cls.pretrained.bcr2gene_decoder.state_dict(),
            "vocab": cls.vocab.vocab, "gene_dim": 3,
            "gene_order": ["A", "B", "C"],
            "bcr_vocab_size": len(cls.vocab.vocab), "max_length": 8,
        }
        torch.save(cls.checkpoint, cls.path)
        cls.obs = pd.DataFrame({
            "CDR3l": ["ACD", "EFG", "HIK", "ACE", "DFG", "HIA"],
            "CDR3h": ["HIK", "ACD", "EFG", "HIA", "ACE", "DFG"],
            "split": ["train", "train", "val", "val", "test", "test"],
        }, index=[f"cell_{i}" for i in range(6)])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def predict(self, adata, **kwargs):
        return predict_regressor(adata, self.path, device="cpu", batch_size=2, **kwargs)

    def test_matches_pretraining_and_freezes_parameters(self):
        model, vocab = load_zeroshot_regressor(self.path)
        light = torch.tensor([vocab.encode(s, 8, True) for s in self.obs.CDR3l])
        heavy = torch.tensor([vocab.encode(s, 8, True) for s in self.obs.CDR3h])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            _, light_features, heavy_features, _ = self.pretrained.foundation(
                torch.randn(6, 3), light, heavy,
            )
            expected = self.pretrained.bcr2gene_decoder(
                torch.cat([light_features, heavy_features], dim=1),
            )
            actual = model(light, heavy)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_bcr_only_input_and_rna_independence(self):
        adata = ad.AnnData(obs=self.obs.copy())
        results, output = self.predict(adata)
        self.assertEqual(results, {"bcr_only": {}})
        self.assertEqual(output.n_vars, 0)
        self.assertEqual(len(adata.obsm), 0)
        key = "X_gene_pred_bcr_only_zeroshot"
        self.assertEqual(output.obsm[key].shape, (6, 3))
        self.assertEqual(output.uns["bcr2gene_zeroshot"]["gene_order"], ["A", "B", "C"])
        with_rna = ad.AnnData(np.full((6, 3), 100.0), obs=self.obs.copy())
        _, another = self.predict(with_rna)
        np.testing.assert_array_equal(output.obsm[key], another.obsm[key])
        output.write_h5ad(Path(self.directory.name) / "predictions.h5ad")

    def test_evaluation_aligns_genes_and_excludes_missing_genes(self):
        # C, an unrelated gene, and A: B must not become an artificial zero target.
        values = np.arange(18, dtype=np.float32).reshape(6, 3)
        adata = ad.AnnData(sparse.csr_matrix(values), obs=self.obs.copy(),
                           var=pd.DataFrame(index=["C", "extra", "A"]))
        splits = ([0, 1], [2, 3], [4, 5])
        results, output = self.predict(adata, evaluate=True, custom_splits=splits,
                                       zero_weight=0.25)
        predicted = output.obsm["X_gene_pred_bcr_only_zeroshot"]
        for group, indices in zip(("train", "val", "test"), splits):
            truth = values[np.ix_(indices, [2, 0])]
            pred = predicted[np.ix_(indices, [0, 2])]
            metrics = results["bcr_only"][group]
            self.assertEqual(metrics["n_genes"], 2)
            self.assertAlmostEqual(metrics["mse"], mean_squared_error(truth, pred), places=4)
            self.assertAlmostEqual(metrics["r2"], r2_score(truth, pred), places=4)
            weights = np.where(truth > 0, 1.0, 0.25)
            expected_loss = (weights * (truth.astype(float) - pred) ** 2).sum() / weights.sum()
            self.assertAlmostEqual(metrics["loss"], expected_loss)
        np.testing.assert_array_equal(output.X.toarray(), values)
        pd.testing.assert_frame_equal(output.obs, self.obs)
        self.assertEqual(list(output.var_names), ["C", "extra", "A"])

    def test_legacy_checkpoint_is_rejected(self):
        checkpoint = dict(self.checkpoint)
        del checkpoint["bcr2gene_decoder_state_dict"]
        path = Path(self.directory.name) / "legacy.pt"
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "bcr2gene_decoder_state_dict"):
            load_zeroshot_regressor(path)

    def test_evaluation_requires_measured_overlap(self):
        adata = ad.AnnData(np.ones((6, 1)), obs=self.obs.copy(),
                           var=pd.DataFrame(index=["unrelated"]))
        with self.assertRaisesRegex(ValueError, "No measured genes overlap"):
            self.predict(adata, evaluate=True)

    def test_overlapping_splits_are_rejected(self):
        adata = ad.AnnData(obs=self.obs.copy())
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.predict(adata, evaluate=True, custom_splits=([0, 1], [1, 2], [4, 5]))


if __name__ == "__main__":
    unittest.main()
