import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .pretrain import align_adata_to_gene_order, clean_bcr_sequence, validate_bcr_columns


class AnnDataEmbeddingDataset(Dataset):
    def __init__(self, adata, bcr_vocab, max_length=30, cdr3l_col="CDR3l", cdr3h_col="CDR3h"):
        validate_bcr_columns(adata, cdr3l_col, cdr3h_col)
        self.adata = adata
        self.bcr_vocab = bcr_vocab
        self.max_length = max_length
        self.cdr3l_col = cdr3l_col
        self.cdr3h_col = cdr3h_col
        self.is_sparse = hasattr(adata.X, "toarray")

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        if self.is_sparse:
            genes = self.adata.X[idx].toarray().squeeze(0)
        else:
            genes = self.adata.X[idx]

        cdr3l = clean_bcr_sequence(self.adata.obs[self.cdr3l_col].iloc[idx])
        cdr3h = clean_bcr_sequence(self.adata.obs[self.cdr3h_col].iloc[idx])

        return {
            "genes": torch.as_tensor(genes, dtype=torch.float32),
            "cdr3l": torch.as_tensor(
                self.bcr_vocab.encode(
                    cdr3l,
                    max_length=self.max_length,
                    add_special_tokens=True,
                ),
                dtype=torch.long,
            ),
            "cdr3h": torch.as_tensor(
                self.bcr_vocab.encode(
                    cdr3h,
                    max_length=self.max_length,
                    add_special_tokens=True,
                ),
                dtype=torch.long,
            ),
        }


def embed_anndata(
    pretrained_model,
    adata,
    bcr_vocab=None,
    batch_size=1024,
    max_length=30,
    device=None,
    copy=True,
    num_workers=4,
    cdr3l_col="CDR3l",
    cdr3h_col="CDR3h",
    gene_order=None,
):
    """
    Extract foundation-model embeddings for an AnnData object.

    Parameters
    ----------
    pretrained_model
        Either a FoundationModel, a fine-tuned wrapper with `.foundation`,
        or a `(model, bcr_vocab)` tuple returned by `load_foundation_model`.
    adata
        AnnData with `.X` gene expression and BCR sequence columns.
    bcr_vocab
        BCRVocab used by the pretrained model. Optional if `pretrained_model`
        is a `(model, vocab)` tuple.
    """

    if isinstance(pretrained_model, tuple):
        pretrained_model, bcr_vocab = pretrained_model

    if bcr_vocab is None:
        raise ValueError(
            "bcr_vocab is required unless pretrained_model is a (model, vocab) tuple."
        )

    validate_bcr_columns(adata, cdr3l_col, cdr3h_col)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    foundation = getattr(pretrained_model, "foundation", pretrained_model)
    gene_order = gene_order if gene_order is not None else getattr(foundation, "gene_order", None)
    adata_out = align_adata_to_gene_order(adata, gene_order, copy=copy)

    was_training = foundation.training
    foundation = foundation.to(device)
    foundation.eval()

    dataset = AnnDataEmbeddingDataset(
        adata_out,
        bcr_vocab,
        max_length=max_length,
        cdr3l_col=cdr3l_col,
        cdr3h_col=cdr3h_col,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    gene_embs = []
    bcr_l_embs = []
    bcr_h_embs = []
    fused_embs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding AnnData"):
            genes = batch["genes"].to(device, non_blocking=True)
            cdr3l = batch["cdr3l"].to(device, non_blocking=True)
            cdr3h = batch["cdr3h"].to(device, non_blocking=True)

            gene_feat, bcr_l_feat, bcr_h_feat, fused_feat = foundation(
                genes,
                cdr3l,
                cdr3h,
            )

            gene_embs.append(gene_feat.cpu().numpy())
            bcr_l_embs.append(bcr_l_feat.cpu().numpy())
            bcr_h_embs.append(bcr_h_feat.cpu().numpy())
            fused_embs.append(fused_feat.cpu().numpy())

    adata_out.obsm["X_gene_emb"] = np.concatenate(gene_embs, axis=0)
    adata_out.obsm["X_bcr_l_emb"] = np.concatenate(bcr_l_embs, axis=0)
    adata_out.obsm["X_bcr_h_emb"] = np.concatenate(bcr_h_embs, axis=0)
    adata_out.obsm["X_bcr_emb"] = np.concatenate(
        [adata_out.obsm["X_bcr_l_emb"], adata_out.obsm["X_bcr_h_emb"]],
        axis=1,
    )
    adata_out.obsm["X_fused_emb"] = np.concatenate(fused_embs, axis=0)

    if was_training:
        foundation.train()

    return adata_out
