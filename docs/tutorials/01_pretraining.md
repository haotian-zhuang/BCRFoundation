# Pretraining BCRFoundation

Train a foundation model using the settings from the scPAIRB pretraining
analysis. If you already have a compatible pretrained checkpoint, you can start
with a downstream tutorial instead.

First complete [installation](../installation.md) and review the
[data and checkpoint requirements](../data_and_checkpoints.md). Run the code
from the repository root and replace the example input path with your data file.

## Prepare the input

Gene expression profiles were library-size-normalized and log-transformed,
followed by selection of highly variable genes. The input below is already
normalized and log-transformed; it is not normalized again in this tutorial.

The analysis uses an existing `adata.var["highly_variable"]` annotation when
available. Otherwise, it selects 3,000 highly variable genes with the Seurat
flavor. The paired CDR3 amino-acid sequences are stored in
`adata.obs["IR_VJ_1_junction_aa"]` and `adata.obs["IR_VDJ_1_junction_aa"]`.

```python
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import BCRFoundation as bcrf

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

cdr3l_col = "IR_VJ_1_junction_aa"
cdr3h_col = "IR_VDJ_1_junction_aa"
adata = sc.read_h5ad("data/pretrain/merged_processed.h5ad")

if "highly_variable" not in adata.var.columns:
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
adata = adata[:, adata.var.highly_variable].copy()
```

## Train the model

This uses 500 epochs, a batch size of 2,048, four gradient-accumulation steps,
and a 10% validation split. Early stopping can end training before 500 epochs.
The RNA, BCR, contrastive, and BCR-to-RNA loss weights are all set to 1.0.

```python
foundation_model, history = bcrf.pretrain.train(
    adata,
    epochs=500,
    save_dir="models/pretraining",
    batch_size=2048,
    accumulation_steps=4,
    val_ratio=0.1,
    early_stopping_patience=50,
    max_length=30,
    add_special_tokens=True,
    warmup_epochs=10,
    improvement_threshold=1e-4,
    gene_loss_weight=1.0,
    bcr_loss_weight=1.0,
    contrastive_alpha=1.0,
    bcr2gene_loss_weight=1.0,
    cdr3l_col=cdr3l_col,
    cdr3h_col=cdr3h_col,
)
```

The package selects CUDA when available and otherwise uses the CPU. The code
above launches a full training run.

## Inspect the outputs

The best checkpoint is saved to
`models/pretraining/foundation_model_best.pt`. Its gene order follows the HVG
columns used for training. Save the loss history for inspection:

```python
loss_history = pd.DataFrame(history)
loss_history.to_csv("models/pretraining/pretrain_loss.csv", index=True)
print(loss_history.tail())
```

To use this new checkpoint in [B cell characteristics prediction](02_classification.md)
or [zero-shot BCR-to-RNA prediction](03_BCR2RNA.md), set that tutorial's
`checkpoint_path` to `"models/pretraining/foundation_model_best.pt"`.
