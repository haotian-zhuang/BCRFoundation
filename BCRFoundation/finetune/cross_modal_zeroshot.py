"""Predict RNA from paired BCR chains without updating pretrained weights."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from ..pretrain import BCRVocab, FoundationModel, validate_bcr_columns
from .cross_modal import _validate_custom_splits


class BCR2GeneZeroShotRegressor(nn.Module):
    """Frozen BCR encoders, normalization layers, and dedicated RNA decoder."""

    def __init__(self, foundation_model, gene_decoder):
        super().__init__()
        self.bcr_encoder_l = foundation_model.bcr_encoder_l
        self.bcr_encoder_h = foundation_model.bcr_encoder_h
        self.norm_bcr_l = foundation_model.norm_bcr_l
        self.norm_bcr_h = foundation_model.norm_bcr_h
        self.gene_decoder = gene_decoder
        self.requires_grad_(False)
        self.eval()

    def forward(self, cdr3l, cdr3h):
        # Match FoundationModel.forward and MaskedPretrainingModel.forward.
        bcr_l_feat = self.norm_bcr_l(self.bcr_encoder_l(cdr3l))
        bcr_h_feat = self.norm_bcr_h(self.bcr_encoder_h(cdr3h))
        return self.gene_decoder(torch.cat([bcr_l_feat, bcr_h_feat], dim=1))


class BCR2GeneZeroShotDataset(Dataset):
    """Tokenize BCR sequences without reading or requiring adata.X."""

    def __init__(self, adata, bcr_vocab, max_length=30, add_special_tokens=True,
                 cdr3l_col="CDR3l", cdr3h_col="CDR3h"):
        validate_bcr_columns(adata, cdr3l_col, cdr3h_col)
        self.cdr3l = adata.obs[cdr3l_col].to_numpy()
        self.cdr3h = adata.obs[cdr3h_col].to_numpy()
        self.bcr_vocab = bcr_vocab
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.cdr3l)

    def __getitem__(self, idx):
        return {
            key: torch.tensor(
                self.bcr_vocab.encode(
                    sequence, max_length=self.max_length,
                    add_special_tokens=self.add_special_tokens,
                ),
                dtype=torch.long,
            )
            for key, sequence in (("cdr3l", self.cdr3l[idx]),
                                  ("cdr3h", self.cdr3h[idx]))
        }


def load_zeroshot_regressor(checkpoint_path):
    """Load a pretraining checkpoint on CPU; reject legacy RNA-only decoders.

    Returns the frozen model and BCR vocabulary. The model exposes
    ``gene_order`` and ``max_length`` from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "bcr2gene_decoder_state_dict" not in checkpoint:
        raise ValueError(
            "Zero-shot prediction requires bcr2gene_decoder_state_dict. "
            "The legacy fused gene decoder is not a trained BCR-to-RNA mapping."
        )
    gene_order = checkpoint.get("gene_order")
    if (gene_order is None or len(gene_order) != checkpoint["gene_dim"]
            or len(set(gene_order)) != len(gene_order)):
        raise ValueError("Checkpoint must contain a unique gene_order matching gene_dim")

    bcr_vocab = BCRVocab()
    bcr_vocab.vocab = checkpoint["vocab"]
    bcr_vocab.inverse_vocab = {v: k for k, v in bcr_vocab.vocab.items()}
    for name in ("pad", "unk", "start", "end"):
        setattr(bcr_vocab, f"{name}_token_idx", bcr_vocab.vocab[f"<{name}>"])

    foundation = FoundationModel(
        gene_dim=checkpoint["gene_dim"],
        bcr_vocab_size=checkpoint["bcr_vocab_size"],
        max_length=checkpoint.get("max_length", 30),
        dropout=0.1,
        pad_idx=bcr_vocab.pad_token_idx,
    )
    foundation.load_state_dict(checkpoint["model_state_dict"])
    decoder = nn.Sequential(
        nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(),
        nn.Linear(512, checkpoint["gene_dim"]),
    )
    decoder.load_state_dict(checkpoint["bcr2gene_decoder_state_dict"])
    model = BCR2GeneZeroShotRegressor(foundation, decoder)
    model.gene_order = list(gene_order)
    model.max_length = checkpoint.get("max_length", 30)
    print(f"Loaded zero-shot BCR2gene model with {len(gene_order)} genes")
    return model, bcr_vocab


def _evaluate_predictions(adata, predictions, indices, target_columns,
                          prediction_columns, zero_weight):
    """Score only measured genes, in checkpoint order, for one set of cells."""
    targets = adata[indices, target_columns].X
    if hasattr(targets, "toarray"):
        targets = targets.toarray()
    targets = np.asarray(targets, dtype=np.float64)
    predicted = predictions[np.ix_(indices, prediction_columns)].astype(np.float64)
    weights = np.where(targets > 0, 1.0, zero_weight)
    weight_sum = weights.sum()
    loss = (float((weights * (predicted - targets) ** 2).sum() / weight_sum)
            if weight_sum > 0 else float("nan"))
    return {
        "loss": loss,
        "mse": float(mean_squared_error(targets, predicted)),
        "r2": float(r2_score(targets, predicted)) if len(indices) >= 2 else float("nan"),
        "n_cells": int(len(indices)),
        "n_genes": int(len(target_columns)),
    }


def predict_regressor(adata, checkpoint_path, batch_size=1024,
                      save_predictions=True, evaluate=False, zero_weight=1.0,
                      cdr3l_col="CDR3l", cdr3h_col="CDR3h",
                      custom_splits=None, device=None, num_workers=0,
                      add_special_tokens=True, copy=True):
    """Run paired-chain zero-shot BCR-to-RNA prediction.

    Args:
        adata: AnnData with paired BCR columns in obs. X is optional unless
            evaluate=True; expression never enters the prediction model.
        checkpoint_path: Pretraining checkpoint containing the dedicated
            bcr2gene decoder, normalization weights, vocabulary, and gene_order.
        batch_size: Number of cells per inference batch.
        save_predictions: Store predictions in the returned AnnData.
        evaluate: Score measured RNA in X, on the same expression scale used
            for pretraining. Missing genes are excluded, never zero-filled.
        zero_weight: Nonnegative weight for nonpositive expression targets in
            evaluation loss, matching cross_modal.weighted_mse_loss. MSE and
            R-squared remain unweighted; R-squared averages across genes.
        cdr3l_col, cdr3h_col: Light- and heavy-chain sequence columns in obs.
        custom_splits: Optional (train, val, test) positional index arrays for
            evaluation, with the same contract as train_regressor. These are
            reporting groups only; no training or model selection takes place.
            Without them, evaluation reports a single "all" group.
        device: Torch device; defaults to CUDA when available, otherwise CPU.
        num_workers: DataLoader worker count; defaults to in-process loading.
        add_special_tokens: Match pretraining tokenization. Defaults to True
            for the project checkpoint; older checkpoints do not record this.
        copy: Copy the input before storing results (default True).

    Returns:
        (results, adata_out), following cross_modal.train_regressor. Results
        contains {"bcr_only": {group: metrics}} (empty groups if evaluate=False).
        Predictions are in obsm["X_gene_pred_bcr_only_zeroshot"], with columns
        named by uns["bcr2gene_zeroshot"]["gene_order"]. Input var, X, observation
        order, and split columns are preserved. Outputs are neither clipped nor
        inverse-transformed. Only paired-chain inference is supported because
        it reproduces the BCR-to-gene pretraining objective.
    """
    validate_bcr_columns(adata, cdr3l_col, cdr3h_col)
    if adata.n_obs == 0:
        raise ValueError("At least one cell is required for prediction")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if not np.isfinite(zero_weight) or zero_weight < 0:
        raise ValueError("zero_weight must be finite and nonnegative")
    if custom_splits is not None and not evaluate:
        raise ValueError("custom_splits requires evaluate=True")
    groups = {"all": np.arange(adata.n_obs)}
    if custom_splits is not None:
        groups = dict(zip(
            ("train", "val", "test"),
            _validate_custom_splits(custom_splits, adata.n_obs),
        ))

    model, bcr_vocab = load_zeroshot_regressor(checkpoint_path)
    if evaluate:
        if adata.X is None or not adata.var_names.is_unique:
            raise ValueError("Evaluation requires X and unique measured gene names")
        source_positions = adata.var_names.get_indexer(model.gene_order)
        prediction_columns = np.flatnonzero(source_positions >= 0)
        target_columns = source_positions[prediction_columns]
        if len(target_columns) == 0:
            raise ValueError("No measured genes overlap the checkpoint gene_order")
        print(f"Evaluating {len(target_columns)} measured genes out of {len(model.gene_order)}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    dataset = BCR2GeneZeroShotDataset(
        adata, bcr_vocab, max_length=model.max_length,
        add_special_tokens=add_special_tokens,
        cdr3l_col=cdr3l_col, cdr3h_col=cdr3h_col,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions = np.empty((adata.n_obs, len(model.gene_order)), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Zero-shot BCR2gene prediction"):
            batch = {key: value.to(device) for key, value in batch.items()}
            predicted = model(batch["cdr3l"], batch["cdr3h"]).cpu().numpy()
            predictions[offset:offset + len(predicted)] = predicted
            offset += len(predicted)

    results = {"bcr_only": {}}
    if evaluate:
        for group, indices in groups.items():
            results["bcr_only"][group] = _evaluate_predictions(
                adata, predictions, indices, target_columns,
                prediction_columns, zero_weight,
            )

    adata_out = adata.copy() if copy else adata
    prediction_key = "X_gene_pred_bcr_only_zeroshot"
    if save_predictions:
        adata_out.obsm[prediction_key] = predictions
        adata_out.uns["bcr2gene_zeroshot"] = {
            "gene_order": model.gene_order,
            "checkpoint_path": str(checkpoint_path),
            "prediction_key": prediction_key,
            "max_length": model.max_length,
            "add_special_tokens": bool(add_special_tokens),
        }
        print(f"Predictions saved to adata.obsm['{prediction_key}']")
    return results, adata_out
