"""Classification dispatch tests; the model-training routine is mocked."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd

from BCRFoundation.finetune import classification


class ClassificationLabelTests(unittest.TestCase):
    labels = ["tissue_name", "disease_name", "celltype_name", "receptor_subtype"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.save_dir = str(Path(self.directory.name) / "models")
        obs = pd.DataFrame(
            {
                "tissue_name": ["PBMC", "Ileum", "Rectum"] * 4,
                "disease_name": ["healthy", "disease"] * 6,
                "celltype_name": ["naive", "memory"] * 6,
                "receptor_subtype": ["IGK", "IGL"] * 6,
                "split_by_sample": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
                "split": ["original"] * 12,
                "CDR3l": ["ACD"] * 12,
                "CDR3h": ["EFG"] * 12,
            },
            index=[f"cell_{i}" for i in range(12)],
        )
        for position, label in enumerate(self.labels):
            obs.loc[f"cell_{position}", label] = None
        self.adata = ad.AnnData(
            np.arange(24, dtype=np.float32).reshape(12, 2),
            obs=obs,
            var=pd.DataFrame(index=["gene_a", "gene_b"]),
        )
        self.adata.obsm["existing"] = np.ones((12, 1))
        self.calls = []
        self.signature = inspect.signature(classification._train_single_classifier)
        self.trainer = patch.object(
            classification, "_train_single_classifier", side_effect=self.fake_train
        ).start()
        self.addCleanup(patch.stopall)

    def fake_train(self, *args, **kwargs):
        options = self.signature.bind(*args, **kwargs)
        options.apply_defaults()
        options = dict(options.arguments)
        self.calls.append(options)
        adata = options["adata"]
        label = options["label_column"]
        slug = label.lower().replace(" ", "_")
        result = adata.copy()
        if options["save_splits"]:
            result.obs["split"] = "unused"
            splits = options["custom_splits"]
            if splits is not None:
                labeled_names = adata.obs_names[adata.obs[label].notna()]
                for name, indices in zip(("train", "val", "test"), splits):
                    result.obs.loc[labeled_names[indices], "split"] = name
        if options["embeddings"]:
            result.obs[f"{slug}_rna_only_pred"] = f"prediction_{label}"
            result.obsm[f"X_{slug}_rna_only_emb"] = np.full((adata.n_obs, 2), len(self.calls))
        metrics = {
            "rna_only": {
                "train": (0.9, 0.8, 0.85),
                "val": (0.8, 0.7, 0.75),
                "test": (0.7, 0.6, 0.65),
            }
        }
        return metrics, result

    def run_labels(self, labels, **kwargs):
        return classification.train_classifier(
            self.adata,
            label_column=labels,
            checkpoint_path="unused_checkpoint.pt",
            modalities=["rna_only"],
            save_dir=self.save_dir,
            **kwargs,
        )

    def test_four_targets_keep_separate_splits_and_combined_predictions(self):
        original = self.adata.copy()
        results, output = self.run_labels(
            self.labels, split_column="split_by_sample", save_splits=True
        )
        self.assertEqual(list(results), self.labels)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len({call["save_dir"] for call in self.calls}), 4)
        for number, (label, call) in enumerate(zip(self.labels, self.calls), 1):
            self.assertEqual(call["label_column"], label)
            pd.testing.assert_frame_equal(call["adata"].obs, original.obs)
            split_values = original.obs.loc[original.obs[label].notna(), "split_by_sample"]
            for group, actual in zip(("train", "val", "test"), call["custom_splits"]):
                np.testing.assert_array_equal(actual, np.flatnonzero(split_values == group))
            expected = original.obs["split_by_sample"].where(
                original.obs[label].notna(), "unused"
            )
            pd.testing.assert_series_equal(output.obs[f"{label}_split"], expected, check_names=False)
            self.assertEqual(output.obs[f"{label}_rna_only_pred"].iloc[0], f"prediction_{label}")
            np.testing.assert_array_equal(output.obsm[f"X_{label}_rna_only_emb"], number)
            self.assertEqual(results[label]["rna_only"]["test"], (0.7, 0.6, 0.65))
        # Tissue labels are already prepared by the caller: the API must not recode them.
        pd.testing.assert_series_equal(output.obs["tissue_name"], original.obs["tissue_name"])
        pd.testing.assert_series_equal(output.obs["split"], original.obs["split"])
        pd.testing.assert_frame_equal(self.adata.obs, original.obs)
        np.testing.assert_array_equal(self.adata.X, original.X)
        self.assertEqual(list(self.adata.obsm), ["existing"])
        np.testing.assert_array_equal(output.obsm["existing"], original.obsm["existing"])

    def test_accepts_common_vector_containers(self):
        for labels in (tuple(self.labels), np.array(self.labels), pd.Index(self.labels)):
            with self.subTest(container=type(labels).__name__):
                self.calls.clear()
                results, _ = self.run_labels(labels, split_column="split_by_sample", save_splits=False)
                self.assertEqual(list(results), self.labels)
                self.assertEqual(len(self.calls), 4)

    def test_string_preserves_flat_results_and_positional_options(self):
        splits = (np.array([0, 1, 2]), np.array([3, 4, 5, 6]), np.array([7, 8, 9, 10]))
        results, _ = classification.train_classifier(
            self.adata, "disease_name", "unused_checkpoint.pt", 3, 2,
            ["rna_only"], False, splits, False, True, "CDR3l", "CDR3h", self.save_dir,
        )
        self.assertEqual(list(results), ["rna_only"])
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["save_dir"], self.save_dir)
        self.assertEqual((call["num_epochs"], call["batch_size"]), (3, 2))
        self.assertFalse(call["embeddings"])
        self.assertFalse(call["save_splits"])
        self.assertTrue(call["return_training_history"])
        for expected, actual in zip(splits, call["custom_splits"]):
            np.testing.assert_array_equal(actual, expected)

    def test_string_default_split_behavior_is_unchanged(self):
        results, _ = self.run_labels("disease_name")
        self.assertEqual(list(results), ["rna_only"])
        self.assertIsNone(self.calls[0]["custom_splits"])
        self.assertEqual(self.calls[0]["num_epochs"], 50)
        self.assertEqual(self.calls[0]["batch_size"], 64)

    def test_string_preserves_existing_unicode_column_names(self):
        label = "病状"
        self.adata.obs[label] = self.adata.obs["disease_name"]
        results, _ = self.run_labels(label, save_splits=False)
        self.assertEqual(list(results), ["rna_only"])
        self.assertEqual(self.calls[0]["label_column"], label)
        self.assertEqual(self.calls[0]["save_dir"], self.save_dir)

    def test_invalid_vectors_fail_before_training(self):
        invalid = ([], [""], [1], ["disease_name", "disease_name"], np.array([self.labels]))
        for labels in invalid:
            with self.subTest(labels=labels):
                with self.assertRaises((TypeError, ValueError)):
                    self.run_labels(labels)
                self.trainer.assert_not_called()
        self.adata.obs["Disease Name"] = self.adata.obs["disease_name"]
        with self.assertRaises(ValueError):
            self.run_labels(["Disease Name", "disease_name"])
        self.trainer.assert_not_called()
        with self.assertRaises((KeyError, ValueError)):
            self.run_labels(["tissue_name", "absent_column"])
        self.trainer.assert_not_called()

    def test_invalid_sample_splits_fail_before_any_task(self):
        for invalid in (None, "unknown"):
            with self.subTest(value=invalid):
                self.adata.obs.loc["cell_0", "split_by_sample"] = invalid
                with self.assertRaises(ValueError):
                    self.run_labels(self.labels, split_column="split_by_sample")
                self.trainer.assert_not_called()
        # This row is unlabeled only for tissue; its split is irrelevant to that task.
        self.run_labels(["tissue_name"], split_column="split_by_sample", save_splits=False)
        self.assertEqual(len(self.calls), 1)
        self.trainer.reset_mock()
        self.adata.obs["split_by_sample"] = "train"
        with self.assertRaises(ValueError):
            self.run_labels(self.labels, split_column="split_by_sample")
        self.trainer.assert_not_called()

    def test_rejects_ambiguous_split_arguments(self):
        splits = ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10])
        with self.assertRaises(ValueError):
            self.run_labels(self.labels, custom_splits=splits, split_column="split_by_sample")
        self.trainer.assert_not_called()
        with self.assertRaises(ValueError):
            self.run_labels(self.labels, custom_splits=splits)
        self.trainer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
