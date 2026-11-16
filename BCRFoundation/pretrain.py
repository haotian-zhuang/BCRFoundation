import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import scanpy as sc
from tqdm import tqdm
import os
import time
import math
from copy import deepcopy
import matplotlib.pyplot as plt

# ----------------------------
# 1. Data Preprocessing
# ----------------------------
def clean_bcr_sequence(seq):
    """Convert missing BCR sequence values to an empty string before tokenization."""
    if seq is None:
        return ""
    seq = str(seq)
    if seq.lower() in {"nan", "none", "<na>"}:
        return ""
    return seq


def validate_bcr_columns(adata, cdr3l_col="CDR3l", cdr3h_col="CDR3h"):
    """Validate that an AnnData object has the requested BCR sequence columns."""
    missing = [col for col in (cdr3l_col, cdr3h_col) if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"adata.obs is missing required BCR columns: {missing}")


def align_adata_to_gene_order(adata, gene_order, copy=True):
    """Align AnnData gene expression to the pretrained gene order, zero-filling missing genes."""
    if gene_order is None:
        return adata.copy() if copy else adata

    gene_order = list(gene_order)
    current_genes = list(adata.var_names)

    if current_genes == gene_order:
        return adata.copy() if copy else adata

    missing_genes = [gene for gene in gene_order if gene not in adata.var_names]
    extra_genes = [gene for gene in current_genes if gene not in gene_order]

    if missing_genes:
        print(
            f"Missing {len(missing_genes)} genes from pretrained gene_order; "
            "filling them with zeros:"
        )
        print(missing_genes)
    if extra_genes:
        print(f"Dropping {len(extra_genes)} genes not present in pretrained gene_order.")

    import anndata as ad
    import pandas as pd
    from scipy import sparse

    gene_to_idx = {gene: idx for idx, gene in enumerate(current_genes)}
    source_positions = np.asarray(
        [gene_to_idx.get(gene, -1) for gene in gene_order], dtype=np.int64
    )
    present_mask = source_positions >= 0

    if sparse.issparse(adata.X):
        if present_mask.all():
            # Selecting every requested column at once avoids repeatedly scanning
            # a row-oriented CSR matrix for each individual gene.
            aligned_X = adata.X[:, source_positions].tocsr()
        else:
            # Select all available genes together in target order, then remap
            # their CSR column indices around the requested all-zero columns.
            target_positions = np.flatnonzero(present_mask)
            selected_X = adata.X[:, source_positions[present_mask]].tocsr()
            aligned_X = sparse.csr_matrix(
                (
                    selected_X.data,
                    target_positions[selected_X.indices],
                    selected_X.indptr,
                ),
                shape=(adata.n_obs, len(gene_order)),
                dtype=adata.X.dtype,
            )
        aligned_X.sort_indices()
    else:
        aligned_X = np.zeros((adata.n_obs, len(gene_order)), dtype=adata.X.dtype)
        target_positions = np.flatnonzero(present_mask)
        aligned_X[:, target_positions] = adata.X[:, source_positions[present_mask]]

    var = adata.var.reindex(gene_order).copy()
    var.index = pd.Index(gene_order, name=adata.var_names.name)
    aligned = ad.AnnData(X=aligned_X, obs=adata.obs.copy(), var=var)
    aligned.uns = adata.uns.copy()
    for key in adata.obsm.keys():
        # Awkward arrays (for example scirpy AIRR data) have no .copy() method.
        aligned.obsm[key] = deepcopy(adata.obsm[key])
    return aligned


class BCRVocab:
    """BCR amino acid sequence vocabulary with extended special tokens."""
    def __init__(self):
        self.amino_acids = set()
        self.special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        self.pad_token_idx = 0
        self.unk_token_idx = 1
        self.start_token_idx = 2
        self.end_token_idx = 3
        self.vocab = {}
        self.inverse_vocab = {}
    
    def build(self, sequences):
        for seq in sequences:
            seq = clean_bcr_sequence(seq)
            self.amino_acids.update(seq)
        self.vocab = {aa: i + len(self.special_tokens) for i, aa in enumerate(sorted(self.amino_acids))}
        for i, tok in enumerate(self.special_tokens):
            self.vocab[tok] = i
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, seq, max_length=30, add_special_tokens=False):
        seq = clean_bcr_sequence(seq)
        if add_special_tokens:
            content_max_length = max_length - 2
            encoded = [self.vocab.get(aa, self.unk_token_idx) for aa in seq[:content_max_length]]
            encoded = [self.start_token_idx] + encoded + [self.end_token_idx]
        else:
            encoded = [self.vocab.get(aa, self.unk_token_idx) for aa in seq[:max_length]]
        if len(encoded) < max_length:
            encoded += [self.pad_token_idx] * (max_length - len(encoded))
        return encoded[:max_length]
    
    def decode(self, tokens, remove_special=True):
        if remove_special:
            tokens = [t for t in tokens if t >= len(self.special_tokens)]
            return ''.join([self.inverse_vocab.get(t, '') for t in tokens])
        else:
            return ''.join([self.inverse_vocab.get(t, '') for t in tokens])

class SingleCellDataset(Dataset):
    def __init__(self, adata, max_length=30, add_special_tokens=False, keep_sparse=True,
                 cdr3l_col="CDR3l", cdr3h_col="CDR3h"):
        validate_bcr_columns(adata, cdr3l_col, cdr3h_col)
        self.adata = adata
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.cdr3l_col = cdr3l_col
        self.cdr3h_col = cdr3h_col
        
        # Option to keep data sparse until needed
        self.is_sparse = keep_sparse and hasattr(adata.X, 'toarray')
        
        if self.is_sparse:
            # Keep sparse matrix, convert on-the-fly in __getitem__
            self.gene_expr = adata.X
        else:
            # Convert to dense tensor upfront
            self.gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            self.gene_expr = torch.FloatTensor(self.gene_expr)
        
        # Build vocabulary
        self.bcr_vocab = BCRVocab()
        all_sequences = list(adata.obs[self.cdr3l_col]) + list(adata.obs[self.cdr3h_col])
        self.bcr_vocab.build(all_sequences)
        
        # Cache the dataset size
        self._len = self.adata.n_obs
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return self._len
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        # Convert sparse to dense on-the-fly if needed
        if self.is_sparse:
            # Extract row and convert to dense
            genes = self.gene_expr[idx].toarray().squeeze()
            genes = torch.FloatTensor(genes)
        else:
            genes = self.gene_expr[idx]
            
        # Encode BCR sequences
        cdr3l = self.bcr_vocab.encode(
            self.adata.obs[self.cdr3l_col].iloc[idx],
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens
        )
        cdr3h = self.bcr_vocab.encode(
            self.adata.obs[self.cdr3h_col].iloc[idx],
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens
        )
        
        return {
            'genes': genes,
            'cdr3l': torch.LongTensor(cdr3l),
            'cdr3h': torch.LongTensor(cdr3h)
        }

# ----------------------------
# 2. Model Architecture
# ----------------------------
class BCRTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, 
                 max_length=30, output_dim=128, dropout=0.1, pad_idx=0):
        super().__init__()
        self.padding_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(max_length + 1, d_model) * 0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.max_length = max_length

    def forward(self, x):
        batch_size = x.size(0)
        
        # Create padding mask (True where padding tokens are)
        padding_mask = (x == self.padding_idx)
        
        token_embeddings = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, token_embeddings], dim=1)
        x = x + self.pos_embedding.unsqueeze(0)
        x = self.embedding_dropout(x)
        x = x.transpose(0, 1)
        
        # Add CLS token position to padding mask
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
        src_key_padding_mask = torch.cat([cls_padding, padding_mask], dim=1)
        
        # Pass mask to transformer
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = x[0]
        out = self.fc(cls_output)
        out = self.layer_norm(out)
        return out

class BCRTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, d_model, max_length, vocab_size, nhead=4, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        
        # Project latent to d_model
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # Learnable query embeddings for each position
        self.query_embeddings = nn.Parameter(torch.randn(max_length, d_model) * 0.02)
        
        # Single transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Smaller FFN
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # Memory from latent
        memory = self.latent_proj(z).unsqueeze(1)  # [batch, 1, d_model]
        
        # Queries for each position
        tgt = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, max_length, d_model]
        
        # Single layer decoding
        out = self.decoder_layer(tgt, memory)
        
        # Project to vocabulary
        logits = self.fc_out(out)  # [batch, max_length, vocab_size]
        
        return logits

class SimpleGeneEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initial projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project input to hidden dimension
        x = self.input_projection(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_projection(x)
        
        return x
        
class FoundationModel(nn.Module):
    def __init__(self, 
                 gene_dim, 
                 bcr_vocab_size, 
                 transformer_d_model=256, 
                 bcr_output_dim=128,
                 hidden_dim=512, 
                 foundation_output_dim=256, 
                 max_length=30, 
                 dropout=0.1,
                 pad_idx=0):
        super().__init__()
        self.gene_encoder = SimpleGeneEncoder(
            input_dim=gene_dim,
            hidden_dim=hidden_dim,
            output_dim=bcr_output_dim,
            dropout=dropout
        )
        
        self.bcr_encoder_l = BCRTransformerEncoder(
            vocab_size=bcr_vocab_size, 
            d_model=transformer_d_model, 
            nhead=8, 
            num_layers=3, 
            max_length=max_length, 
            output_dim=bcr_output_dim, 
            dropout=dropout, 
            pad_idx=pad_idx
        )
        self.bcr_encoder_h = BCRTransformerEncoder(
            vocab_size=bcr_vocab_size, 
            d_model=transformer_d_model, 
            nhead=8, 
            num_layers=3, 
            max_length=max_length, 
            output_dim=bcr_output_dim, 
            dropout=dropout, 
            pad_idx=pad_idx
        )
        self.norm_gene = nn.LayerNorm(bcr_output_dim)
        self.norm_bcr_l = nn.LayerNorm(bcr_output_dim)
        self.norm_bcr_h = nn.LayerNorm(bcr_output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(bcr_output_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, foundation_output_dim),
            nn.LayerNorm(foundation_output_dim)
        )
        
    def forward(self, genes, cdr3l, cdr3h):
        gene_feat = self.gene_encoder(genes)
        bcr_l_feat = self.bcr_encoder_l(cdr3l)
        bcr_h_feat = self.bcr_encoder_h(cdr3h)
        gene_feat = self.norm_gene(gene_feat)
        bcr_l_feat = self.norm_bcr_l(bcr_l_feat)
        bcr_h_feat = self.norm_bcr_h(bcr_h_feat)
        combined = torch.cat([gene_feat, bcr_l_feat, bcr_h_feat], dim=1)
        fused_feat = self.fusion(combined)
        return gene_feat, bcr_l_feat, bcr_h_feat, fused_feat

# ----------------------------
# 3. Self-supervised Pretraining
# ----------------------------
def contrastive_loss(featuresA, featuresB, temperature=0.07):
    A = F.normalize(featuresA, dim=1)
    B = F.normalize(featuresB, dim=1)
    sim_matrix = torch.matmul(A, B.t()) / temperature
    labels = torch.arange(A.size(0), device=A.device)
    loss_AB = F.cross_entropy(sim_matrix, labels)
    loss_BA = F.cross_entropy(sim_matrix.t(), labels)
    return 0.5 * (loss_AB + loss_BA)

class MaskedPretrainingModel(nn.Module):
    def __init__(self, foundation_model, gene_dim, max_length, bcr_vocab_size,
                 gene_loss_weight=1.0, bcr_loss_weight=1.0, contrastive_alpha=1.0, bcr2gene_loss_weight=1.0):
        super().__init__()
        self.foundation = foundation_model
        self.gene_loss_weight = gene_loss_weight
        self.bcr_loss_weight = bcr_loss_weight
        self.contrastive_alpha = contrastive_alpha
        self.bcr2gene_loss_weight = bcr2gene_loss_weight
        
        self.gene_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, gene_dim)
        )
        self.gene_mask_token = nn.Parameter(torch.randn(gene_dim) * 0.02)
        
        self.bcr_decoder_l = BCRTransformerDecoder(
            latent_dim=256,
            d_model=256,
            max_length=max_length,
            vocab_size=bcr_vocab_size,
            nhead=4,  # Use default
            dropout=0.1
        )
        self.bcr_decoder_h = BCRTransformerDecoder(
            latent_dim=256,
            d_model=256,
            max_length=max_length,
            vocab_size=bcr_vocab_size,
            nhead=4,  # Use default
            dropout=0.1
        )

        self.bcr2gene_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, gene_dim)
        )
        
    def forward(self, batch, gene_mask_ratio=0.35):
        batch_size = batch['genes'].shape[0]
        mask = torch.rand(batch['genes'].shape, device=batch['genes'].device) < gene_mask_ratio
        masked_genes = batch['genes'].clone()
        
        # Use learned mask token instead of zeros
        mask_token_expanded = self.gene_mask_token.unsqueeze(0).expand(batch_size, -1)
        masked_genes[mask] = mask_token_expanded.expand_as(masked_genes)[mask]
        gene_feat, bcr_l_feat, bcr_h_feat, fused_feat = self.foundation(
            masked_genes, 
            batch['cdr3l'], 
            batch['cdr3h']
        )
        gene_recon = self.gene_decoder(fused_feat)
        gene_loss = F.mse_loss(gene_recon[mask], batch['genes'][mask])
        logits_l = self.bcr_decoder_l(fused_feat)
        logits_h = self.bcr_decoder_h(fused_feat)
        vocab_size = logits_l.size(-1)
        logits_l_flat = logits_l.view(-1, vocab_size)
        logits_h_flat = logits_h.view(-1, vocab_size)
        targets_l = batch['cdr3l'].view(-1)
        targets_h = batch['cdr3h'].view(-1)
        bcr_loss_l = F.cross_entropy(logits_l_flat, targets_l, ignore_index=0)
        bcr_loss_h = F.cross_entropy(logits_h_flat, targets_h, ignore_index=0)
        bcr_loss = 0.5 * (bcr_loss_l + bcr_loss_h)
        c_loss = 0.5 * contrastive_loss(gene_feat, bcr_l_feat, temperature=0.07) + \
                 0.5 * contrastive_loss(gene_feat, bcr_h_feat, temperature=0.07)
        
        bcr_only_feat = torch.cat([bcr_l_feat, bcr_h_feat], dim=1)
        bcr2gene_recon = self.bcr2gene_decoder(bcr_only_feat)
        bcr2gene_loss = F.mse_loss(bcr2gene_recon, batch['genes'])

        total_loss = self.gene_loss_weight * gene_loss + self.bcr_loss_weight * bcr_loss + self.contrastive_alpha * c_loss + self.bcr2gene_loss_weight * bcr2gene_loss
        return total_loss, gene_loss, bcr_loss, c_loss, bcr2gene_loss
        
def load_foundation_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = FoundationModel(
        gene_dim=checkpoint['gene_dim'],
        bcr_vocab_size=checkpoint['bcr_vocab_size'],
        max_length=checkpoint.get('max_length', 30)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.gene_order = checkpoint.get('gene_order')
    return model, checkpoint['vocab']
    
# ----------------------------
# Revised Training Loop
# ----------------------------
def train(adata, epochs=100, batch_size=2048, save_dir='models', 
          accumulation_steps=4, val_ratio=0.1, early_stopping_patience=50,
          max_length=30, add_special_tokens=True,
          warmup_epochs=10, improvement_threshold=1e-4,
          gene_loss_weight=1.0, bcr_loss_weight=1.0, contrastive_alpha=1.0, bcr2gene_loss_weight=1.0,
          cdr3l_col="CDR3l", cdr3h_col="CDR3h"):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    os.makedirs(save_dir, exist_ok=True)
    dataset = SingleCellDataset(
        adata, 
        max_length=max_length, 
        add_special_tokens=add_special_tokens,
        cdr3l_col=cdr3l_col,
        cdr3h_col=cdr3h_col
    )
    
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    foundation = FoundationModel(
        gene_dim=adata.n_vars,
        bcr_vocab_size=len(dataset.bcr_vocab.vocab),
        max_length=max_length,
        dropout=0.1
    )
    model = MaskedPretrainingModel(
        foundation,
        gene_dim=adata.n_vars,
        max_length=max_length,
        bcr_vocab_size=len(dataset.bcr_vocab.vocab),
        gene_loss_weight=gene_loss_weight,
        bcr_loss_weight=bcr_loss_weight,
        contrastive_alpha=contrastive_alpha,
        bcr2gene_loss_weight=bcr2gene_loss_weight
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Combined LR scheduler with warmup and cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            # cosine decay after warmup
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    gradient_clip_norm = 1.0  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Using device: {device}")
    
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        'train_loss': [],
        'val_loss': [],
        'gene_loss': [],
        'bcr_loss': [],
        'contrastive_loss': [],
        'bcr2gene_loss': []
    }
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training Progress")
    first_epoch_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_gene_loss = 0.0
        epoch_bcr_loss = 0.0
        epoch_c_loss = 0.0
        epoch_bcr2gene_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, gene_loss, bcr_loss, c_loss, bcr2gene_loss = model(batch)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * accumulation_steps
            epoch_gene_loss += gene_loss.item()
            epoch_bcr_loss += bcr_loss.item()
            epoch_c_loss += c_loss.item()
            epoch_bcr2gene_loss += bcr2gene_loss.item()
            pbar.set_postfix({'Loss': f'{loss.item() * accumulation_steps:.4f}'})
            pbar.update(1)
        
        if epoch == 0:
            first_epoch_end_time = time.time()
            print(f"Time for first epoch: {first_epoch_end_time - first_epoch_start_time:.2f} seconds")

        avg_train_loss = epoch_loss / len(train_loader)
        avg_gene_loss = epoch_gene_loss / len(train_loader)
        avg_bcr_loss = epoch_bcr_loss / len(train_loader)
        avg_c_loss = epoch_c_loss / len(train_loader)
        avg_bcr2gene_loss = epoch_bcr2gene_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, _, _, _, _ = model(batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['gene_loss'].append(avg_gene_loss)
        history['bcr_loss'].append(avg_bcr_loss)
        history['contrastive_loss'].append(avg_c_loss)
        history['bcr2gene_loss'].append(avg_bcr2gene_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Gene Loss: {avg_gene_loss:.4f}, '
                  f'BCR Loss: {avg_bcr_loss:.4f}, '
                  f'Contrastive Loss: {avg_c_loss:.4f}, '
                  f'BCR2Gene Loss: {avg_bcr2gene_loss:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # Check for significant improvement
        if best_val_loss - avg_val_loss > improvement_threshold:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            checkpoint_path = os.path.join(save_dir, 'foundation_model_best.pt')
            checkpoint = {
                'model_state_dict': model.foundation.state_dict(),
                'gene_decoder_state_dict': model.gene_decoder.state_dict(),
                'bcr_decoder_l_state_dict': model.bcr_decoder_l.state_dict(),
                'bcr_decoder_h_state_dict': model.bcr_decoder_h.state_dict(),
                'bcr2gene_decoder_state_dict': model.bcr2gene_decoder.state_dict(),
                'vocab': dataset.bcr_vocab.vocab,
                'epoch': epoch + 1,
                'loss': avg_val_loss,
                'gene_dim': adata.n_vars,
                'gene_order': adata.var_names.tolist(),
                'bcr_vocab_size': len(dataset.bcr_vocab.vocab),
                'max_length': max_length,
                'special_tokens': dataset.bcr_vocab.special_tokens
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best epoch: {best_epoch}")
                break

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f'foundation_model_epoch_{epoch+1}.pt')
            checkpoint = {
                'model_state_dict': model.foundation.state_dict(),
                'gene_decoder_state_dict': model.gene_decoder.state_dict(),
                'bcr_decoder_l_state_dict': model.bcr_decoder_l.state_dict(),
                'bcr_decoder_h_state_dict': model.bcr_decoder_h.state_dict(),
                'bcr2gene_decoder_state_dict': model.bcr2gene_decoder.state_dict(),
                'vocab': dataset.bcr_vocab.vocab,
                'epoch': epoch + 1,
                'loss': avg_train_loss,
                'gene_dim': adata.n_vars,
                'gene_order': adata.var_names.tolist(),
                'bcr_vocab_size': len(dataset.bcr_vocab.vocab),
                'max_length': max_length,
                'special_tokens': dataset.bcr_vocab.special_tokens
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

    pbar.close()

    best_model_path = os.path.join(save_dir, 'foundation_model_best.pt')
    best_model, _ = load_foundation_model(best_model_path)
    return best_model, history
