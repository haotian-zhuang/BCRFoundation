# Tutorial 2: Cell-Type Classification

Fine-tune BCRFoundation using the cell-type classification settings from the
IBD-Boland analysis. This tutorial compares RNA-only, paired-BCR, individual
BCR-chain, and combined RNA–BCR representations.

First complete [installation](../installation.md) and prepare the inputs
listed in [data and checkpoints](../data_and_checkpoints.md). Run the code from
the repository root. The example expects a compatible checkpoint at
`pretrain_model/foundation_model_best.pt`; change that path if using a model
from [pretraining](01_pretraining.md).

## Load the processed data

The input must contain normalized, log-transformed expression in `adata.X`,
paired CDR3 amino-acid columns, cell-type labels in `celltype_name`, and saved
sample-based split labels in `split_by_sample`.

```python
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import BCRFoundation as bcrf
from BCRFoundation.finetune.utils import build_results_dataframe

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cdr3l_col = "IR_VJ_1_junction_aa"
cdr3h_col = "IR_VDJ_1_junction_aa"
label_column = "celltype_name"
split_column = "split_by_sample"
checkpoint_path = "pretrain_model/foundation_model_best.pt"

adata = sc.read_h5ad("data/downstream/IBD-Boland/merged_processed_split.h5ad")
```

## Use the saved sample-based splits

The classifier removes cells with missing labels before applying the split
indices. Construct positional indices in that same labeled-cell order.
All three groups must be represented, with no missing or unknown split values.

```python
required_columns = {label_column, split_column, cdr3l_col, cdr3h_col}
missing_columns = required_columns.difference(adata.obs.columns)
if missing_columns:
    raise KeyError(f"Missing adata.obs columns: {sorted(missing_columns)}")

split_values = adata.obs.loc[
    adata.obs[label_column].notna(), split_column
].astype("string")
if split_values.isna().any() or set(split_values.unique()) != {"train", "val", "test"}:
    raise ValueError("Labeled cells must have nonmissing train/val/test split values")

split_array = split_values.to_numpy()
custom_splits = tuple(
    np.flatnonzero(split_array == name) for name in ("train", "val", "test")
)
print(dict(zip(("train", "val", "test"), map(len, custom_splits))))
```

## Fine-tune the classifiers

The analysis trains each modality for 50 epochs with a batch size of 512.
`save_splits=False` retains the existing split annotations.

```python
results, adata_classified = bcrf.finetune.classification.train_classifier(
    adata,
    label_column=label_column,
    checkpoint_path=checkpoint_path,
    num_epochs=50,
    batch_size=512,
    modalities=["rna_only", "bcr_only", "bcrl_only", "bcrh_only", "rna_bcr"],
    custom_splits=custom_splits,
    save_splits=False,
    cdr3l_col=cdr3l_col,
    cdr3h_col=cdr3h_col,
    save_dir="models/IBD-Boland/classification/celltype",
)
```

The package aligns expression genes to the checkpoint's gene order and selects
the best classifier for each modality using validation accuracy.

## Inspect and save the results

Each split's metrics are returned as accuracy, macro F1, and weighted F1.
The returned AnnData includes the classification embeddings and predictions.

```python
for modality, metrics in results.items():
    accuracy, f1_macro, f1_weighted = metrics["test"]
    print(
        f"{modality}: accuracy={accuracy:.3f}, "
        f"macro F1={f1_macro:.3f}, weighted F1={f1_weighted:.3f}"
    )

output_dir = Path("outputs/IBD-Boland/classification/celltype")
output_dir.mkdir(parents=True, exist_ok=True)
build_results_dataframe(results).to_csv(output_dir / "celltype_results.csv", index=False)
adata_classified.write_h5ad(output_dir / "adata_with_celltype_predictions.h5ad")
```

For [zero-shot RNA evaluation](03_BCR2RNA.md), reload the original processed
input as shown in that tutorial. Classification aligns and fills missing gene
columns, so its returned expression matrix should not be used as the measured
RNA reference for that evaluation.
