# Zero-Shot BCR-to-RNA Prediction

Predict gene expression from paired BCR sequences using frozen pretrained
weights. This reproduces the IBD-Boland zero-shot analysis: evaluate all cells
together, without using train/validation/test split annotations.

First complete [installation](../installation.md) and review
[data and checkpoints](../data_and_checkpoints.md). Run the code from the
repository root. A compatible checkpoint must contain the dedicated BCR-to-RNA
decoder, BCR vocabulary, and pretrained gene order.

## Load the checkpoint path and processed input

The example expects the checkpoint at
`pretrain_model/foundation_model_best.pt`. To use your own newly trained model,
change the path to the output from [pretraining](01_pretraining.md).

Reload the original processed data rather than reusing the classification
output. Measured RNA in `adata.X` is used for evaluation only; it is not a
prediction input. Its expression scale must match the normalized,
log-transformed expression used for pretraining.

```python
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import BCRFoundation as bcrf

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cdr3l_col = "IR_VJ_1_junction_aa"
cdr3h_col = "IR_VDJ_1_junction_aa"
checkpoint_path = "pretrain_model/foundation_model_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adata = sc.read_h5ad("data/downstream/IBD-Boland/merged_processed_split.h5ad")
required_columns = {cdr3l_col, cdr3h_col}
missing_columns = required_columns.difference(adata.obs.columns)
if missing_columns:
    raise KeyError(f"Missing adata.obs columns: {sorted(missing_columns)}")
if not adata.obs_names.is_unique:
    raise ValueError("AnnData observation names must be unique")
```

## Predict and evaluate all cells

The analysis uses batches of 512, saves predictions, and evaluates against the
measured genes that overlap the checkpoint's gene order. No `custom_splits`
argument is supplied. With `copy=False`, predictions are added to the loaded
AnnData object.

```python
results, adata_pred = bcrf.finetune.cross_modal_zeroshot.predict_regressor(
    adata,
    checkpoint_path=checkpoint_path,
    batch_size=512,
    evaluate=True,
    save_predictions=True,
    zero_weight=1.0,
    cdr3l_col=cdr3l_col,
    cdr3h_col=cdr3h_col,
    device=device,
    num_workers=0,
    add_special_tokens=True,
    copy=False,
)
```

For BCR-only data without measured RNA, set `evaluate=False`. Both CDR3 columns
are still required, but expression and split labels are not needed for
prediction. In that case, `results["bcr_only"]` is empty.

## Inspect and save the outputs

Prediction columns follow the checkpoint's gene order, not `adata.var_names`.
The returned values are on the training expression scale; they are not converted
back to raw counts. Evaluation excludes genes that were not measured in the
input.

```python
predictions = adata_pred.obsm["X_gene_pred_bcr_only_zeroshot"]
gene_names = adata_pred.uns["bcr2gene_zeroshot"]["gene_order"]
metrics = results["bcr_only"].get("all")

print(predictions.shape)
if metrics is not None:
    print(metrics)  # loss, mse, r2, n_cells, n_genes

output_dir = Path("outputs/IBD-Boland/BCR2RNA_zeroshot")
output_dir.mkdir(parents=True, exist_ok=True)
adata_pred.write_h5ad(output_dir / "adata_with_predictions.h5ad")
if metrics is not None:
    pd.DataFrame([
        {"modality": "bcr_only", "group": "all", **metrics}
    ]).to_csv(output_dir / "BCR2gene_summary.csv", index=False)
```

The `"all"` metrics above are available with `evaluate=True`, as used in this
tutorial. They do not involve fitting or selecting a model on the evaluation
cells.
