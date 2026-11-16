"""Batch-adversarial integration with masked gene reconstruction.

RNA-only, RNA--BCR, and BCR-only support alternating encoder/classifier updates.
BCR-only embeddings use sequences alone; measured RNA remains a training target.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from ..pretrain import align_adata_to_gene_order
from .classification import (
    RepresentationDataset,
    SingleCellClassificationDataset,
    load_foundation_model,
)


MODALITIES = ('rna_bcr', 'rna_only', 'bcr_only')
DEFAULT_MODALITIES = ('rna_bcr', 'rna_only', 'bcr_only')
CHECKPOINT_VERSION = 3


class GradientReversal(torch.autograd.Function):
    """Identity forward pass and gradient reversal with multiplier one."""

    @staticmethod
    def forward(ctx, embedding):
        return embedding.view_as(embedding)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def _load_gene_decoder(checkpoint):
    if 'gene_decoder_state_dict' not in checkpoint:
        raise ValueError('Integration requires a pretrained gene_decoder_state_dict.')
    decoder = nn.Sequential(
        nn.Linear(256, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Linear(512, checkpoint['gene_dim']),
    )
    decoder.load_state_dict(checkpoint['gene_decoder_state_dict'])
    return decoder


class BatchIntegrationModel(nn.Module):
    """Shared encoders, a gene reconstruction branch, and an MLP batch classifier.

    Export normalized RNA features (128 dimensions), fused RNA--BCR features
    (256), or concatenated normalized light/heavy BCR features (256). Only the
    batch branch crosses gradient reversal. Dropout follows train()/eval().
    """

    def __init__(self, foundation_model, checkpoint, num_classes, mode='rna_bcr',
                 observed_gene_mask=None, classifier_hidden=128):
        super().__init__()
        if mode not in MODALITIES:
            raise ValueError(f'Unknown integration modality: {mode}')
        if num_classes < 2 or classifier_hidden < 0:
            raise ValueError('Require at least two batches and a nonnegative hidden width.')
        self.foundation = foundation_model
        self.mode = mode
        self.embedding_dim = 128 if mode == 'rna_only' else 256
        if observed_gene_mask is None:
            observed_gene_mask = torch.ones(checkpoint['gene_dim'], dtype=torch.bool)
        observed_gene_mask = torch.as_tensor(observed_gene_mask, dtype=torch.bool).clone()
        if observed_gene_mask.shape != (checkpoint['gene_dim'],) or not observed_gene_mask.any():
            raise ValueError('observed_gene_mask must select genes in checkpoint order.')
        self.register_buffer('observed_gene_mask', observed_gene_mask)
        self.gene_decoder = _load_gene_decoder(checkpoint)

        # Initialize the linear head before the mask token, then construct the
        # MLP without advancing the shared masking/dropout random-number stream.
        # This keeps that stream consistent across classifier architectures.
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        token = checkpoint.get('gene_mask_token')
        if token is None:
            warnings.warn('Pretrained mask token unavailable; initializing N(0, 0.02^2).')
            token = torch.randn(checkpoint['gene_dim']) * 0.02
        self.gene_mask_token = nn.Parameter(token.detach().clone())
        self.foundation.requires_grad_(False)
        modules = [self.foundation.fusion]
        if mode != 'bcr_only':
            modules.extend((self.foundation.gene_encoder, self.foundation.norm_gene))
        else:
            self.gene_mask_token.requires_grad_(False)
        if mode != 'rna_only':
            modules.extend((self.foundation.bcr_encoder_l, self.foundation.norm_bcr_l,
                            self.foundation.bcr_encoder_h, self.foundation.norm_bcr_h))
        for module in modules:
            module.requires_grad_(True)
        if classifier_hidden:
            with torch.random.fork_rng():
                self.classifier = nn.Sequential(
                    nn.Linear(self.embedding_dim, classifier_hidden),
                    nn.GELU(),
                    nn.Linear(classifier_hidden, num_classes),
                )

    def get_embedding(self, genes, cdr3l, cdr3h):
        if self.mode == 'rna_bcr':
            return self.foundation(genes, cdr3l, cdr3h)[3]
        if self.mode == 'bcr_only':
            light = self.foundation.norm_bcr_l(self.foundation.bcr_encoder_l(cdr3l))
            heavy = self.foundation.norm_bcr_h(self.foundation.bcr_encoder_h(cdr3h))
            return torch.cat((light, heavy), dim=1)
        return self.foundation.norm_gene(self.foundation.gene_encoder(genes))

    def reconstruct_genes(self, embedding):
        if self.mode == 'rna_only':
            zeros = torch.zeros_like(embedding)
            latent = self.foundation.fusion(torch.cat((embedding, zeros, zeros), dim=1))
        elif self.mode == 'bcr_only':
            zeros = embedding.new_zeros((embedding.shape[0], 128))
            latent = self.foundation.fusion(torch.cat((zeros, embedding), dim=1))
        else:
            latent = embedding
        return self.gene_decoder(latent)

    def masked_embedding(self, batch, gene_mask_ratio=0.35, generator=None):
        genes = batch['genes']
        mask = (torch.rand(genes.shape, device=genes.device, generator=generator)
                < gene_mask_ratio) & self.observed_gene_mask.unsqueeze(0)
        if not mask.any():
            mask[0, self.observed_gene_mask.nonzero(as_tuple=True)[0][0]] = True
        masked = torch.where(mask, self.gene_mask_token.unsqueeze(0), genes)
        # BCR-only deliberately ignores masked RNA; mask selects MSE targets.
        embedding = self.get_embedding(masked, batch['cdr3l'], batch['cdr3h'])
        return embedding, mask

    def forward(self, batch, gene_mask_ratio=0.35, generator=None):
        embedding, mask = self.masked_embedding(batch, gene_mask_ratio, generator)
        prediction = self.reconstruct_genes(embedding)
        gene_loss = F.mse_loss(prediction[mask], batch['genes'][mask])
        unmasked = self.get_embedding(batch['genes'], batch['cdr3l'], batch['cdr3h'])
        logits = self.classifier(GradientReversal.apply(unmasked))
        return gene_loss, logits


def _build_optimizers(model, encoder_lr=1e-5, decoder_lr=1e-5, classifier_lr=1e-3):
    groups = {'encoder': [], 'decoder': [], 'classifier': []}
    names = {key: [] for key in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('classifier.'):
            group = 'classifier'
        elif name.startswith(('gene_decoder.', 'foundation.fusion.')):
            group = 'decoder'
        else:
            group = 'encoder'
        groups[group].append(parameter)
        names[group].append(name)
    encoder_parameters = groups['encoder'] + groups['decoder']
    encoder_optimizer = torch.optim.Adam([
        dict(params=groups['encoder'], lr=encoder_lr),
        dict(params=groups['decoder'], lr=decoder_lr),
    ], weight_decay=0.0)
    classifier_optimizer = torch.optim.Adam(
        groups['classifier'], lr=classifier_lr, weight_decay=0.0,
    )
    return encoder_optimizer, classifier_optimizer, encoder_parameters, names


def _train_batch(model, batch, encoder_optimizer, classifier_optimizer,
                 encoder_parameters, classifier_steps=20, batch_loss_weight=0.01,
                 gene_mask_ratio=0.35, gene_loss_weight=1.0, classifier_loss_scale=0.01):
    encoder_optimizer.zero_grad(set_to_none=True)
    classifier_optimizer.zero_grad(set_to_none=True)
    embedding, mask = model.masked_embedding(batch, gene_mask_ratio)
    unmasked = model.get_embedding(batch['genes'], batch['cdr3l'], batch['cdr3h'])
    with torch.no_grad():
        logits_before = model.classifier(unmasked.detach())
        ce_before = F.cross_entropy(logits_before, batch['label'])
        acc_before = (logits_before.argmax(1) == batch['label']).float().mean()
    for _ in range(classifier_steps):
        classifier_optimizer.zero_grad(set_to_none=True)
        ce = F.cross_entropy(model.classifier(unmasked.detach()), batch['label'])
        if not torch.isfinite(ce):
            raise FloatingPointError('Non-finite batch-classification loss.')
        (classifier_loss_scale * ce).backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
        classifier_optimizer.step()
    classifier_optimizer.zero_grad(set_to_none=True)
    model.classifier.requires_grad_(False)
    try:
        # Preserve dropout ordering: reconstruction follows unmasked encoding
        # and classifier updates; reuse the original unmasked encoder graph.
        prediction = model.reconstruct_genes(embedding)
        gene_loss = F.mse_loss(prediction[mask], batch['genes'][mask])
        logits = model.classifier(GradientReversal.apply(unmasked))
        batch_loss = F.cross_entropy(logits, batch['label'])
        total = gene_loss_weight * gene_loss + batch_loss_weight * batch_loss
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite integration loss.')
        total.backward()
        torch.nn.utils.clip_grad_norm_(encoder_parameters, 1.0)
        encoder_optimizer.step()
    finally:
        model.classifier.requires_grad_(True)
    return dict(
        loss=float(total.detach()), gene_loss=float(gene_loss.detach()),
        batch_loss=float(batch_loss.detach()),
        encoder_objective=float((gene_loss_weight * gene_loss - batch_loss_weight * batch_loss).detach()),
        batch_acc=float((logits.argmax(1) == batch['label']).float().mean()),
        batch_loss_before_classifier=float(ce_before),
        batch_acc_before_classifier=float(acc_before),
        batch_loss_improvement_in_classifier_steps=float(ce_before - batch_loss.detach()),
    )


def train_epoch(model, dataloader, encoder_optimizer, classifier_optimizer,
                encoder_parameters, device, classifier_steps=20,
                batch_loss_weight=0.01, gene_mask_ratio=0.35,
                gene_loss_weight=1.0, classifier_loss_scale=0.01):
    """Alternate detached classifier updates and one shared encoder update."""
    model.train()
    totals, count = {}, 0
    for batch in dataloader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        values = _train_batch(
            model, batch, encoder_optimizer, classifier_optimizer, encoder_parameters,
            classifier_steps, batch_loss_weight, gene_mask_ratio, gene_loss_weight,
            classifier_loss_scale,
        )
        size = len(batch['genes'])
        count += size
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + size * value
    return {key: value / count for key, value in totals.items()}


def evaluate(model, dataloader, device, batch_loss_weight=0.01, gene_mask_ratio=0.35,
             gene_loss_weight=1.0):
    """Evaluate using fixed gene masks, unmasked batch inputs, and disabled dropout."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(0)
    totals, count = {}, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            gene_loss, logits = model(batch, gene_mask_ratio, generator)
            batch_loss = F.cross_entropy(logits, batch['label'])
            values = dict(
                loss=float(gene_loss_weight * gene_loss + batch_loss_weight * batch_loss),
                gene_loss=float(gene_loss), batch_loss=float(batch_loss),
                encoder_objective=float(gene_loss_weight * gene_loss - batch_loss_weight * batch_loss),
                batch_acc=float((logits.argmax(1) == batch['label']).float().mean()),
            )
            size = len(batch['genes'])
            count += size
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + size * value
    return {key: value / count for key, value in totals.items()}


def _with_bcr_columns(adata, modalities, cdr3l_col, cdr3h_col):
    required = (cdr3l_col, cdr3h_col) if any(m != 'rna_only' for m in modalities) else ()
    missing = [column for column in required if column not in adata.obs]
    if missing:
        raise KeyError(f'adata.obs is missing required BCR columns: {missing}')
    prepared = adata.copy()
    for column in (cdr3l_col, cdr3h_col):
        if column not in prepared.obs:
            prepared.obs[column] = ''
    return prepared


def _split_indices(size, custom_splits, val_ratio=0.1):
    """Return sorted train/validation indices with the fixed seed-0 split."""
    if custom_splits is None:
        if not 0 < val_ratio < 1 or size < 2:
            raise ValueError('Require at least two cells and val_ratio strictly between 0 and 1.')
        train_size = int((1 - val_ratio) * size)
        if not 0 < train_size < size:
            raise ValueError('Both training and validation must be nonempty.')
        train, val = random_split(
            range(size), [train_size, size - train_size],
            generator=torch.Generator().manual_seed(0),
        )
        return np.sort(train.indices), np.sort(val.indices)
    if len(custom_splits) != 2:
        raise ValueError('custom_splits must contain train and validation indices.')
    splits = []
    for indices in custom_splits:
        indices = np.asarray(indices)
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu':
            raise ValueError('Each split must be a nonempty one-dimensional integer array.')
        if (indices < 0).any() or (indices >= size).any():
            raise ValueError('Split indices address cells after filtering missing batch labels.')
        splits.append(np.sort(indices))
    combined = np.concatenate(splits)
    if len(np.unique(combined)) != len(combined):
        raise ValueError('Train/validation indices must be unique and disjoint.')
    return tuple(splits)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _export_embeddings(model, dataset, batch_size, device):
    model.eval()
    embeddings, predictions = [], []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(2),
    )
    with torch.no_grad():
        for batch in loader:
            embedding = model.get_embedding(
                batch['genes'].to(device), batch['cdr3l'].to(device), batch['cdr3h'].to(device),
            )
            embeddings.append(embedding.cpu().numpy())
            predictions.append(model.classifier(embedding).argmax(1).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(predictions)


def train_integrator(adata, batch_column: str, checkpoint_path: str,
                     num_epochs: int = 50, batch_size: int = 512,
                     modalities=DEFAULT_MODALITIES, embeddings=True,
                     custom_splits=None, save_splits=True,
                     return_training_history: bool = False,
                     cdr3l_col: str = 'CDR3l', cdr3h_col: str = 'CDR3h',
                     save_dir: Optional[str] = '../fine_tune_models/integration',
                     gene_loss_weight: float = 1.0, batch_loss_weight: float = 0.01,
                     gene_mask_ratio: float = 0.35, num_workers: int = 4,
                     improvement_threshold: float = 0.0, val_ratio: float = 0.1, *,
                     random_state: int = 0, encoder_lr: float = 1e-5,
                     decoder_lr: float = 1e-5, classifier_lr: float = 1e-3,
                     classifier_steps: int = 20, classifier_hidden: int = 128,
                     classifier_loss_scale: float = 0.01,
                     epoch_callback=None) -> Tuple[Dict[str, Any], Any]:
    """Fine-tune independent modes and return (results, aligned AnnData).

    The default order is RNA--BCR, RNA-only, then BCR-only.
    Training combines masked-gene MSE and unmasked batch cross-entropy with
    enabled dropout, unit gradient reversal, and alternating optimizers.
    Each mode starts from the same pretrained checkpoint with its own seeded RNG.
    RNA-only requires expression; RNA--BCR and BCR-only additionally require both
    CDR3 columns. BCR-only uses expression as reconstruction supervision only.

    Missing genes are zero-filled for alignment and excluded from masked MSE.
    Missing batch labels are excluded from fitting but receive embeddings.
    custom_splits=(train_idx, val_idx) addresses the labeled cells. The default
    split uses seed 0 regardless of random_state, and its indices are sorted.

    Train for num_epochs and retain the final epoch; validation never selects a
    checkpoint. Epoch 0 records initialization. Histories contain epoch 0 through
    num_epochs when requested, and are returned in memory. By default, losses
    are printed to the log and embeddings are exported only at the final epoch.
    epoch_callback(mode, epoch, row, compact_adata)
    receives losses, diagnostics, and unmasked embeddings for all cells at every
    epoch. The callback must not mutate its inputs. No scIB scoring is performed
    by this training function; exported epoch embeddings can be scored separately.

    Checkpoints are version-3 dictionaries named last_<batch_column>_integration_
    <mode>.pt. Existing checkpoint files are rejected; callers may archive them
    before training. save_dir=None disables saving. The input AnnData is copied;
    final embeddings are stored as obsm['X_<mode>_emb'] in the returned object.
    improvement_threshold is retained only for callers passing its former default
    zero; there is no early stopping or improvement-based checkpoint selection.
    """
    modalities = tuple(modalities)
    if not modalities or len(set(modalities)) != len(modalities) or any(m not in MODALITIES for m in modalities):
        raise ValueError(f'modalities must be a nonempty, unique selection from {MODALITIES}.')
    for name, value, minimum in [('num_epochs', num_epochs, 1), ('batch_size', batch_size, 1),
                                 ('num_workers', num_workers, 0), ('classifier_steps', classifier_steps, 1),
                                 ('classifier_hidden', classifier_hidden, 0), ('random_state', random_state, 0)]:
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if random_state >= 2 ** 32:
        raise ValueError('random_state must be below 2**32.')
    for name, value in [('encoder_lr', encoder_lr), ('decoder_lr', decoder_lr),
                        ('classifier_lr', classifier_lr), ('gene_loss_weight', gene_loss_weight),
                        ('classifier_loss_scale', classifier_loss_scale)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if not np.isfinite(batch_loss_weight) or batch_loss_weight < 0:
        raise ValueError('batch_loss_weight must be finite and nonnegative.')
    if not 0 < gene_mask_ratio <= 1 or improvement_threshold != 0:
        raise ValueError('Require gene_mask_ratio in (0, 1] and improvement_threshold=0.')
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError('epoch_callback must be callable.')
    if batch_column not in adata.obs:
        raise KeyError(f'adata.obs is missing batch column: {batch_column}')
    if adata.X is None or adata.n_obs < 2 or not adata.n_vars or not adata.var_names.is_unique:
        raise ValueError('Require measured expression, at least two cells, and unique gene names.')
    task_name = str(batch_column).lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
    model_paths = {mode: str(Path(save_dir) / f'last_{task_name}_integration_{mode}.pt')
                   if save_dir is not None else None for mode in modalities}
    for path in model_paths.values():
        if path is not None and Path(path).exists():
            raise FileExistsError(f'Archive the existing integration checkpoint before training: {path}')

    # Configure determinism before the first CUDA operation.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    gene_order = checkpoint.get('gene_order')
    if gene_order is not None and not adata.var_names.isin(gene_order).any():
        raise ValueError('No measured genes overlap the pretrained gene panel.')
    if 'gene_decoder_state_dict' not in checkpoint:
        raise ValueError('Integration requires pretrained gene_decoder_state_dict.')
    observed = (np.asarray([g in adata.var_names for g in gene_order], dtype=bool)
                if gene_order is not None else np.ones(adata.n_vars, dtype=bool))
    aligned = align_adata_to_gene_order(adata.copy(), gene_order)
    if aligned.n_vars != checkpoint['gene_dim']:
        raise ValueError('Aligned expression dimensions differ from the checkpoint.')
    positions = np.flatnonzero(aligned.obs[batch_column].notna().to_numpy())
    labeled = aligned[positions].copy()
    batches = sorted(labeled.obs[batch_column].unique().tolist())
    if len(batches) < 2:
        raise ValueError('Integration requires at least two nonmissing batch labels.')
    batch_encoder = {label: index for index, label in enumerate(batches)}
    foundation, vocab = load_foundation_model(checkpoint_path)
    del foundation
    prepared = _with_bcr_columns(labeled, modalities, cdr3l_col, cdr3h_col)
    dataset = SingleCellClassificationDataset(
        prepared, vocab, batch_encoder, batch_column, max_length=checkpoint.get('max_length', 30),
        cdr3l_col=cdr3l_col, cdr3h_col=cdr3h_col,
    )
    if not torch.isfinite(dataset.gene_expr).all():
        raise ValueError('Reconstruction targets must be finite.')
    train_idx, val_idx = _split_indices(len(dataset), custom_splits, val_ratio)
    if set(np.asarray(dataset.labels)[train_idx]) != set(batch_encoder.values()):
        raise ValueError('Training must contain every batch; supply suitable custom_splits.')
    split_labels = np.full(aligned.n_obs, 'unused', dtype=object)
    for name, indices in [('train', train_idx), ('val', val_idx)]:
        split_labels[positions[indices]] = name
    if save_splits:
        aligned.obs['split'] = split_labels
    export_obs = aligned.obs.copy()
    export_obs['split'] = split_labels
    full_dataset = dataset if len(positions) == aligned.n_obs else RepresentationDataset(
        _with_bcr_columns(aligned, modalities, cdr3l_col, cdr3h_col), vocab,
        max_length=checkpoint.get('max_length', 30), cdr3l_col=cdr3l_col, cdr3h_col=cdr3h_col,
    )
    labels = np.asarray(dataset.labels)
    val_labels = labels[val_idx]
    val_counts = np.bincount(val_labels, minlength=len(batch_encoder))
    print(f'Integration: {aligned.n_obs} cells; {len(batches)} batches; '
          f'{len(train_idx)} training and {len(val_idx)} validation cells; device={device}')
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    for mode in modalities:
        print(f'\nTraining integration: {mode}')
        _seed_everything(random_state)
        foundation, _ = load_foundation_model(checkpoint_path)
        model = BatchIntegrationModel(
            foundation, checkpoint, len(batches), mode, observed, classifier_hidden,
        ).to(device)
        encoder_optimizer, classifier_optimizer, parameters, parameter_names = _build_optimizers(
            model, encoder_lr, decoder_lr, classifier_lr,
        )
        train_generator = torch.Generator().manual_seed(random_state)
        val_generator = torch.Generator().manual_seed(1)
        common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)
        train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,
                                  generator=train_generator, **common)
        val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False,
                                generator=val_generator, **common)
        fixed_train = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=False,
            num_workers=0, generator=torch.Generator().manual_seed(3),
        )
        config = dict(
            mode=mode, seed=random_state, split_seed=0, val_ratio=val_ratio,
            epochs=num_epochs, batch_size=batch_size, gene_loss_weight=gene_loss_weight,
            batch_loss_weight=batch_loss_weight, mask_ratio=gene_mask_ratio,
            encoder_lr=encoder_lr, decoder_lr=decoder_lr, classifier_lr=classifier_lr,
            classifier_steps=classifier_steps, classifier_hidden=classifier_hidden,
            classifier_loss_scale=classifier_loss_scale, adversarial_dropout=True,
            grl_multiplier=1.0, optimizer='Adam', weight_decay=0.0, gradient_clip_norm=1.0,
            checkpoint_selection='last_epoch', parameter_groups=parameter_names,
        )
        history = [] if return_training_history else None
        for epoch in range(num_epochs + 1):
            if epoch == 0:
                with torch.random.fork_rng(devices=devices):
                    train_metrics = evaluate(model, fixed_train, device, batch_loss_weight,
                                             gene_mask_ratio, gene_loss_weight)
                train_metrics.update(batch_loss_before_classifier=train_metrics['batch_loss'],
                                     batch_acc_before_classifier=train_metrics['batch_acc'],
                                     batch_loss_improvement_in_classifier_steps=0.0)
            else:
                train_metrics = train_epoch(
                    model, train_loader, encoder_optimizer, classifier_optimizer,
                    parameters, device, classifier_steps, batch_loss_weight,
                    gene_mask_ratio, gene_loss_weight, classifier_loss_scale,
                )
            val_metrics = evaluate(model, val_loader, device, batch_loss_weight,
                                   gene_mask_ratio, gene_loss_weight)
            row = dict(epoch=epoch, **{f'train_{k}': float(v) for k, v in train_metrics.items()},
                       **{f'val_{k}': float(v) for k, v in val_metrics.items()},
                       encoder_lr_used=encoder_lr, decoder_lr_used=decoder_lr,
                       classifier_lr_used=classifier_lr)
            if epoch == num_epochs or return_training_history or epoch_callback is not None:
                with torch.random.fork_rng(devices=devices):
                    embedding, prediction = _export_embeddings(model, full_dataset, batch_size, device)
                    if not np.isfinite(embedding).all():
                        raise FloatingPointError(f'Non-finite {mode} embeddings at epoch {epoch}.')
                    val_prediction = prediction[positions[val_idx]]
                    row['unmasked_val_batch_accuracy'] = float(np.mean(val_prediction == val_labels))
                    row['unmasked_val_batch_balanced_accuracy'] = float(np.mean([
                        np.mean(val_prediction[val_labels == c] == c)
                        for c in range(len(batches)) if val_counts[c]
                    ]))
                    row['val_majority_accuracy'] = float(val_counts.max() / len(val_labels))
                    if epoch_callback is not None:
                        compact = ad.AnnData(obs=export_obs.copy())
                        compact.obsm[f'X_{mode}_emb'] = embedding.copy()
                        compact.uns['epoch'] = epoch
                        epoch_callback(mode, epoch, dict(row), compact)
            if not all(np.isfinite(value) for value in row.values()):
                raise FloatingPointError(f'Non-finite {mode} diagnostics at epoch {epoch}.')
            if history is not None:
                history.append(row)
            print(f'Mode {mode} Epoch {epoch}/{num_epochs}: '
                  f'Train MSE={row["train_gene_loss"]:.6f} | '
                  f'Val MSE={row["val_gene_loss"]:.6f} | '
                  f'Val batch CE={row["val_batch_loss"]:.6f}', flush=True)
        if embeddings:
            aligned.obsm[f'X_{mode}_emb'] = embedding
        model_path = model_paths[mode]
        if model_path is not None:
            stored = dict(
                format_version=CHECKPOINT_VERSION, mode=mode, epoch=num_epochs,
                state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                config=config, gene_order=gene_order, batch_encoder=batch_encoder,
                pretrained_checkpoint=str(checkpoint_path),
                encoder_optimizer_state_dict=encoder_optimizer.state_dict(),
                classifier_optimizer_state_dict=classifier_optimizer.state_dict(),
                split_labels=split_labels,
                obs_names=aligned.obs_names.to_numpy(), python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                train_loader_rng=train_generator.get_state(), val_loader_rng=val_generator.get_state(),
            )
            with open(model_path, 'xb') as stream:
                torch.save(stored, stream)
            print(f'Saved final integration model: {model_path}')
            del stored
        result = dict(
            train=train_metrics, val=val_metrics, model_path=model_path,
            batch_encoder=batch_encoder.copy(), selected_epoch=num_epochs,
            best_epoch=num_epochs, best_val_loss=val_metrics['gene_loss'],
            embedding_dim=model.embedding_dim, config=config,
            checkpoint_selection='last_epoch', gene_loss_weight=gene_loss_weight,
            batch_loss_weight=batch_loss_weight,
        )
        if return_training_history:
            result['history'] = history
        results[mode] = result
        del model, foundation, encoder_optimizer, classifier_optimizer, parameters, train_loader, val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return results, aligned


def extract_and_store_embeddings(adata, checkpoint_path, model_paths, batch_size=512,
                                 max_length=None, cdr3l_col='CDR3l', cdr3h_col='CDR3h',
                                 num_workers=0):
    """Load integration checkpoints and return all input cells with unmasked embeddings.

    Version-3 checkpoints are required; unsupported checkpoint formats are
    rejected to prevent loading weights into a different architecture. For
    BCR-only, expression may be a zero placeholder on the checkpoint gene panel.
    """
    if not model_paths or any(mode not in MODALITIES for mode in model_paths):
        raise ValueError(f'model_paths must select from {MODALITIES}.')
    if batch_size < 1 or num_workers < 0:
        raise ValueError('Require positive batch_size and nonnegative num_workers.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    foundation, vocab = load_foundation_model(checkpoint_path)
    del foundation
    aligned = align_adata_to_gene_order(adata.copy(), checkpoint.get('gene_order'))
    dataset = RepresentationDataset(
        _with_bcr_columns(aligned, model_paths, cdr3l_col, cdr3h_col), vocab,
        max_length=max_length or checkpoint.get('max_length', 30),
        cdr3l_col=cdr3l_col, cdr3h_col=cdr3h_col,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        generator=torch.Generator().manual_seed(2))
    for mode, path in model_paths.items():
        stored = torch.load(path, map_location='cpu', weights_only=False)
        if stored.get('format_version') != CHECKPOINT_VERSION or stored.get('mode') != mode:
            raise ValueError(f'Expected finalized version-{CHECKPOINT_VERSION} {mode} checkpoint: {path}')
        if stored.get('gene_order') != checkpoint.get('gene_order'):
            raise ValueError('Integration and pretrained checkpoint gene orders differ.')
        foundation, _ = load_foundation_model(checkpoint_path)
        model = BatchIntegrationModel(
            foundation, checkpoint, len(stored['batch_encoder']), mode,
            stored['state_dict']['observed_gene_mask'], stored['config']['classifier_hidden'],
        )
        model.load_state_dict(stored['state_dict'])
        model.to(device).eval()
        chunks = []
        with torch.no_grad():
            for batch in loader:
                chunks.append(model.get_embedding(
                    batch['genes'].to(device), batch['cdr3l'].to(device), batch['cdr3h'].to(device),
                ).cpu().numpy())
        embedding = np.concatenate(chunks)
        if not np.isfinite(embedding).all():
            raise FloatingPointError(f'Non-finite {mode} embeddings.')
        aligned.obsm[f'X_{mode}_emb'] = embedding
        del model, foundation
    return aligned
