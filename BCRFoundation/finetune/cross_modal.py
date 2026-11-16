import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import scanpy as sc
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, List, Any
from ..pretrain import BCRVocab, FoundationModel, align_adata_to_gene_order, validate_bcr_columns

class BCR2GeneRegressor(nn.Module):
    """
    A model that takes BCR sequences and predicts gene expression.
    Uses the pretrained foundation model's BCR encoders and gene decoder.
    Fine-tunes both the BCR encoder and gene decoder.
    """
    def __init__(self, foundation_model, gene_decoder, gene_dim, modality="both",
                 freeze_bcr_encoder=False, use_bcr2gene_decoder=False):
        super().__init__()
        self.modality = modality.lower()
        self.use_bcr2gene_decoder = use_bcr2gene_decoder
        
        # Extract only the needed components from foundation model
        self.bcr_encoder_l = foundation_model.bcr_encoder_l
        self.bcr_encoder_h = foundation_model.bcr_encoder_h
        self.gene_decoder = gene_decoder
        
        # Option to freeze BCR encoder parameters if needed
        if freeze_bcr_encoder:
            for param in self.bcr_encoder_l.parameters():
                param.requires_grad = False
            for param in self.bcr_encoder_h.parameters():
                param.requires_grad = False
            
        # Get the output dimensions from the BCR encoders
        bcr_output_dim = self.bcr_encoder_l.fc.out_features  # Should be 128
        
        # Define modality-specific projection layers
        if self.modality == "bcr_only":
            # Use both BCR chains
            input_dim = bcr_output_dim * 2
        elif self.modality == "bcrl_only":
            # Use only light chain
            input_dim = bcr_output_dim
        elif self.modality == "bcrh_only":
            # Use only heavy chain
            input_dim = bcr_output_dim
        else:
            raise ValueError(f"Unsupported modality: {modality}")
            
        if not self.use_bcr2gene_decoder:
            # Preserve the original path for checkpoints that predate the
            # dedicated BCR-to-gene decoder.
            self.bcr2latent_projection = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
    def forward(self, cdr3l, cdr3h):
        # Get BCR encodings directly from the BCR encoders
        bcr_l_feat = self.bcr_encoder_l(cdr3l)
        bcr_h_feat = self.bcr_encoder_h(cdr3h)
        
        if self.use_bcr2gene_decoder:
            # Match the [light, heavy] feature layout used during pretraining.
            if self.modality == "bcr_only":
                latent = torch.cat([bcr_l_feat, bcr_h_feat], dim=1)
            elif self.modality == "bcrl_only":
                latent = torch.cat([bcr_l_feat, torch.zeros_like(bcr_l_feat)], dim=1)
            elif self.modality == "bcrh_only":
                latent = torch.cat([torch.zeros_like(bcr_h_feat), bcr_h_feat], dim=1)
        else:
            # Preserve the original projection path for older checkpoints.
            if self.modality == "bcr_only":
                embeddings = torch.cat([bcr_l_feat, bcr_h_feat], dim=1)
            elif self.modality == "bcrl_only":
                embeddings = bcr_l_feat
            elif self.modality == "bcrh_only":
                embeddings = bcr_h_feat
            latent = self.bcr2latent_projection(embeddings)
        
        # Decode to gene expression using pretrained decoder
        gene_expression = self.gene_decoder(latent)
        
        return gene_expression

class BCR2GeneDataset(Dataset):
    """Dataset for BCR2Gene regression task."""
    def __init__(self, adata, bcr_vocab, max_length=30, add_special_tokens=True,
                 cdr3l_col="CDR3l", cdr3h_col="CDR3h"):
        validate_bcr_columns(adata, cdr3l_col, cdr3h_col)
        self.adata = adata
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.cdr3l_col = cdr3l_col
        self.cdr3h_col = cdr3h_col
        
        # Extract gene expression
        self.gene_expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        self.gene_expr = torch.FloatTensor(self.gene_expr)
        
        # Use the provided BCR vocabulary
        self.bcr_vocab = bcr_vocab
        
    def __len__(self):
        return self.adata.n_obs
    
    def __getitem__(self, idx):
        genes = self.gene_expr[idx]
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

def load_foundation_model(checkpoint_path: str) -> Tuple[nn.Module, BCRVocab]:
    """
    Load a pretrained foundation model and its vocabulary.
    
    Args:
        checkpoint_path: Path to the model checkpoint

    Returns:
        Tuple containing:
            - The loaded foundation model
            - The BCRVocab object (not just dictionary)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = FoundationModel(
        gene_dim=checkpoint['gene_dim'],
        bcr_vocab_size=checkpoint['bcr_vocab_size'],
        max_length=checkpoint.get('max_length', 30),
        dropout=0.1,
        pad_idx=0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.gene_order = checkpoint.get('gene_order')
    
    # Reconstruct BCRVocab object from checkpoint
    bcr_vocab = BCRVocab()
    bcr_vocab.vocab = checkpoint['vocab']
    bcr_vocab.inverse_vocab = {v: k for k, v in checkpoint['vocab'].items()}
    # Set special token indices based on the vocab
    bcr_vocab.pad_token_idx = checkpoint['vocab']['<pad>']
    bcr_vocab.unk_token_idx = checkpoint['vocab']['<unk>']
    bcr_vocab.start_token_idx = checkpoint['vocab']['<start>']
    bcr_vocab.end_token_idx = checkpoint['vocab']['<end>']
    
    # Get the max_length from the checkpoint
    max_length = checkpoint.get('max_length', 30)
    print(f"Loaded model with max_length: {max_length}")
    
    return model, bcr_vocab

def load_gene_decoder(checkpoint_path):
    """Load the dedicated BCR-to-gene decoder when available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gene_dim = checkpoint['gene_dim']
    use_bcr2gene_decoder = 'bcr2gene_decoder_state_dict' in checkpoint
    
    # Create a wrapper for the gene decoder
    class GeneDecoder(nn.Module):
        def __init__(self, decoder_state_dict):
            super().__init__()
            decoder_layers = [
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, gene_dim)
            ]
            if not use_bcr2gene_decoder:
                decoder_layers.append(nn.GELU())
            self.decoder = nn.Sequential(*decoder_layers)
            self.decoder.load_state_dict(decoder_state_dict)
            
        def forward(self, x):
            return self.decoder(x)
    
    if use_bcr2gene_decoder:
        decoder_state_dict = checkpoint['bcr2gene_decoder_state_dict']
        print("Using pretrained bcr2gene_decoder_state_dict")
    else:
        decoder_state_dict = checkpoint['gene_decoder_state_dict']
        print("bcr2gene_decoder_state_dict not found; using legacy gene_decoder_state_dict path")

    decoder = GeneDecoder(decoder_state_dict)
    return decoder, use_bcr2gene_decoder

def weighted_mse_loss(gene_pred, gene_true, zero_weight=1.0):
    loss_all = F.mse_loss(gene_pred, gene_true, reduction='none')
    weight = torch.where(gene_true > 0, 1.0, zero_weight)
    return (loss_all * weight).sum() / weight.sum()

def evaluate_regressor(model, dataloader, device, zero_weight, description):
    """Evaluate one data split and return loss, MSE, and R-squared."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            batch = {k: v.to(device) for k, v in batch.items()}
            gene_pred = model(batch['cdr3l'], batch['cdr3h'])
            loss = weighted_mse_loss(
                gene_pred,
                batch['genes'],
                zero_weight=zero_weight,
            )
            # loss = F.mse_loss(gene_pred, batch['genes'])
            # loss = F.smooth_l1_loss(gene_pred, batch['genes'])

            total_loss += loss.item() * batch['genes'].size(0)
            predictions.append(gene_pred.cpu().numpy())
            targets.append(batch['genes'].cpu().numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    return {
        'loss': total_loss / len(dataloader.dataset),
        'mse': mean_squared_error(targets, predictions),
        'r2': r2_score(targets, predictions),
    }

def _validate_custom_splits(custom_splits, dataset_size):
    """Validate and normalize positional train/validation/test indices."""
    if not isinstance(custom_splits, (tuple, list)) or len(custom_splits) != 3:
        raise ValueError(
            "custom_splits must contain exactly three index arrays: "
            "(train, val, test)"
        )

    split_names = ("train", "val", "test")
    normalized = []
    for split_name, indices in zip(split_names, custom_splits):
        split_indices = np.asarray(indices)
        if split_indices.ndim != 1:
            raise ValueError(f"{split_name} indices must be one-dimensional")
        if split_indices.size == 0:
            raise ValueError(f"{split_name} indices must not be empty")
        if not np.issubdtype(split_indices.dtype, np.integer):
            raise TypeError(f"{split_name} indices must be integers")

        split_indices = split_indices.astype(np.int64, copy=False)
        if np.any(split_indices < 0) or np.any(split_indices >= dataset_size):
            raise IndexError(
                f"{split_name} indices must be between 0 and {dataset_size - 1}"
            )
        if np.unique(split_indices).size != split_indices.size:
            raise ValueError(f"{split_name} indices contain duplicates")
        normalized.append(split_indices)

    all_indices = np.concatenate(normalized)
    if all_indices.size != dataset_size or np.unique(all_indices).size != dataset_size:
        raise ValueError(
            "custom_splits must be disjoint and assign every dataset observation "
            "exactly once"
        )

    return tuple(normalized)

def train_regressor(adata, checkpoint_path, num_epochs=50, batch_size=1024, 
                    learning_rate=1e-4, modalities=None, val_split=0.1, test_split=0.1,
                    save_splits=True, save_predictions=True, zero_weight=1.0,
                    cdr3l_col="CDR3l", cdr3h_col="CDR3h",
                    save_dir="../fine_tune_models/BCR2gene", custom_splits=None):
    """
    Train a BCR to gene expression regressor using the pretrained foundation model.
    
    Args:
        adata: AnnData object with gene expression and BCR sequences
        checkpoint_path: Path to the pretrained foundation model checkpoint
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        modalities: List of modalities to train ("bcr_only", "bcrl_only", "bcrh_only")
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        save_splits: Whether to save train/val/test splits in adata.obs
        save_predictions: Whether to save predictions in adata.obsm
        custom_splits: Optional positional (train, val, test) index arrays. When
            omitted, preserve the original seeded random splitting behavior.
        
    Returns:
        Dictionary of training results for each modality and updated adata
    """
    if modalities is None:
        modalities = ["bcr_only", "bcrl_only", "bcrh_only"]

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    max_length = checkpoint.get('max_length', 30)
    adata = align_adata_to_gene_order(adata, checkpoint.get('gene_order'))
    
    # Load the vocabulary once; model components are reloaded for each modality
    # so every fine-tuning run starts from the same pretrained weights.
    vocab_source_model, bcr_vocab = load_foundation_model(checkpoint_path)
    del vocab_source_model
    
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = BCR2GeneDataset(
        adata,
        bcr_vocab,
        max_length=max_length,
        cdr3l_col=cdr3l_col,
        cdr3h_col=cdr3h_col,
    )
    
    # Split the dataset. Keep the original random split as the default so
    # existing callers retain exactly the same behavior.
    if custom_splits is None:
        train_size = int((1 - val_split - test_split) * len(dataset))
        val_size = int(val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size

        generator = torch.Generator().manual_seed(0)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
    else:
        train_indices, val_indices, test_indices = _validate_custom_splits(
            custom_splits, len(dataset)
        )
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        print(
            "Using custom splits: "
            f"Train={len(train_dataset)}, "
            f"Val={len(val_dataset)}, "
            f"Test={len(test_dataset)}"
        )
    
    # Record split information in adata if requested
    if save_splits:
        # Create a split column in adata.obs
        adata.obs['split'] = 'train'  # Default all to train
        
        # Update with validation and test indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices
        
        adata.obs.iloc[val_indices, adata.obs.columns.get_loc('split')] = 'val'
        adata.obs.iloc[test_indices, adata.obs.columns.get_loc('split')] = 'test'
        
        print(f"Split information saved in adata.obs['split']:")
        print(f"  - Train: {(adata.obs['split'] == 'train').sum()} cells")
        print(f"  - Val: {(adata.obs['split'] == 'val').sum()} cells")
        print(f"  - Test: {(adata.obs['split'] == 'test').sum()} cells")
    
    # Create data loaders
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Track results for each modality
    results = {}
    
    # Train a model for each modality
    for modality in modalities:
        print(f"\n=== Training {modality} regressor ===")

        # Start each modality from an independent copy of the pretrained weights.
        foundation_copy, _ = load_foundation_model(checkpoint_path)
        gene_decoder, use_bcr2gene_decoder = load_gene_decoder(checkpoint_path)
        foundation_copy = foundation_copy.to(device)
        gene_decoder = gene_decoder.to(device)
        
        # Create the regressor model with fine-tunable components
        regressor = BCR2GeneRegressor(
            foundation_model=foundation_copy,
            gene_decoder=gene_decoder,
            gene_dim=adata.n_vars,
            modality=modality,
            freeze_bcr_encoder=False,  # Allow fine-tuning of BCR encoder
            use_bcr2gene_decoder=use_bcr2gene_decoder
        )
        regressor = regressor.to(device)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(regressor.parameters(), lr=learning_rate)
        
        # Set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Track best validation loss
        best_val_loss = float('inf')
        modality_results = {"train": {}, "val": {}, "test": {}}
        
        # Create directory for saving models
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(
            save_dir,
            f"best_bcr2gene_regressor_{modality}.pt",
        )

        # Training loop
        for epoch in range(num_epochs):
            # Training
            regressor.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                gene_pred = regressor(batch['cdr3l'], batch['cdr3h'])
                
                # Compute loss
                loss = weighted_mse_loss(gene_pred, batch['genes'], zero_weight=zero_weight)
                # loss = F.mse_loss(gene_pred, batch['genes'])
                # loss = F.smooth_l1_loss(gene_pred, batch['genes'])
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * batch['genes'].size(0)
                train_preds.append(gene_pred.detach().cpu().numpy())
                train_targets.append(batch['genes'].detach().cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_dataset)
            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)
            train_mse = mean_squared_error(train_targets, train_preds)
            train_r2 = r2_score(train_targets, train_preds)
            
            # Validation
            regressor.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    gene_pred = regressor(batch['cdr3l'], batch['cdr3h'])
                    
                    # Compute loss
                    loss = weighted_mse_loss(gene_pred, batch['genes'], zero_weight=zero_weight)
                    # loss = F.mse_loss(gene_pred, batch['genes'])
                    
                    # Track metrics
                    val_loss += loss.item() * batch['genes'].size(0)
                    val_preds.append(gene_pred.cpu().numpy())
                    val_targets.append(batch['genes'].cpu().numpy())
            
            # Calculate metrics
            val_loss /= len(val_dataset)
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            val_mse = mean_squared_error(val_targets, val_preds)
            val_r2 = r2_score(val_targets, val_preds)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Train R²: {train_r2:.4f}, "
                  f"Val R²: {val_r2:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': regressor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'modality': modality,
                    'gene_dim': adata.n_vars
                }, best_model_path)
                print(f"Saved best model checkpoint (Val Loss: {val_loss:.6f})")
        
        # Evaluate on test set with best model
        print("\nEvaluating on test set with best model...")
        best_model = BCR2GeneRegressor(
            foundation_model=foundation_copy,
            gene_decoder=gene_decoder,
            gene_dim=adata.n_vars,
            modality=modality,
            freeze_bcr_encoder=False,
            use_bcr2gene_decoder=use_bcr2gene_decoder
        )
        checkpoint = torch.load(best_model_path)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model = best_model.to(device)
        best_model.eval()
        
        # Recompute every reported split with the same best-validation model.
        train_metrics = evaluate_regressor(
            best_model, train_loader, device, zero_weight, "Evaluating best model (Train)"
        )
        val_metrics = evaluate_regressor(
            best_model, val_loader, device, zero_weight, "Evaluating best model (Val)"
        )
        test_metrics = evaluate_regressor(
            best_model, test_loader, device, zero_weight, "Evaluating best model (Test)"
        )

        print(
            f"Test Loss: {test_metrics['loss']:.6f}, "
            f"Test MSE: {test_metrics['mse']:.6f}, "
            f"Test R²: {test_metrics['r2']:.4f}"
        )
        
        # Generate predictions for all data points using the best model
        if save_predictions:
            print(f"\nGenerating predictions for all data with best {modality} model...")
            full_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
            all_preds = []
            
            with torch.no_grad():
                for batch in tqdm(full_loader, desc="Predicting"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    gene_pred = best_model(batch['cdr3l'], batch['cdr3h'])
                    
                    # Store predictions
                    all_preds.append(gene_pred.cpu().numpy())
            
            # Save predictions to adata
            all_preds = np.vstack(all_preds)
            adata.obsm[f'X_gene_pred_{modality}'] = all_preds
            print(f"Predictions saved to adata.obsm['X_gene_pred_{modality}']")
        
        # Store results
        modality_results["train"] = train_metrics
        modality_results["val"] = val_metrics
        modality_results["test"] = test_metrics
        
        results[modality] = modality_results
    
    return results, adata
