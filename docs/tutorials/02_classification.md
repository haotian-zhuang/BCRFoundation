# B cell characteristics prediction

Use BCRFoundation to predict tissue origin, disease status, cell subtype, and
light chain locus type.

| Task | Label column in `adata.obs` |
| --- | --- |
| Tissue origin | `tissue_name` |
| Disease status | `disease_name` |
| Cell subtype | `celltype_name` |
| Light chain locus type | `receptor_subtype` |

The example below predicts **B cell subtype**, using the ready-to-use labels in
`adata.obs["celltype_name"]`. For another task, change `label_column` to the
corresponding column above. It compares combined RNA–BCR, RNA-only, paired-BCR,
and individual BCR-chain representations.

First complete [installation](../installation.md) and review the
[data and checkpoint requirements](../data_and_checkpoints.md). Run the example
from the repository root and replace the input path with your data file.

## Load the data

The input should contain processed gene expression, paired CDR3 amino-acid
sequences, and B cell subtype labels. This example also uses existing `train`,
`val`, and `test` assignments in `adata.obs["split_by_sample"]`.

```python
from pathlib import Path

import numpy as np
import scanpy as sc
import BCRFoundation as bcrf
from BCRFoundation.finetune.utils import build_results_dataframe

adata = sc.read_h5ad("data/downstream/IBD-Boland/merged_processed_split.h5ad")
label_column = "celltype_name"
split_column = "split_by_sample"
```

## Use the saved splits

The classifier excludes cells with missing target labels, so construct the split
indices in that same labeled-cell order.

```python
split_values = adata.obs.loc[
    adata.obs[label_column].notna(), split_column
].astype("string")
if split_values.isna().any() or set(split_values.unique()) != {"train", "val", "test"}:
    raise ValueError("Labeled cells must have nonmissing train/val/test split values")

split_array = split_values.to_numpy()
custom_splits = (
    np.flatnonzero(split_array == "train"),
    np.flatnonzero(split_array == "val"),
    np.flatnonzero(split_array == "test"),
)
```

## Fine-tune the classifiers

Pass one label column name. `save_splits=False` preserves the existing split
annotations. The classifier with the best validation accuracy is selected for
each modality.

```python
results, adata_classified = bcrf.finetune.classification.train_classifier(
    adata,
    label_column=label_column,
    checkpoint_path="pretrain_model/foundation_model_best.pt",
    num_epochs=50,
    batch_size=512,
    modalities=["rna_bcr", "rna_only", "bcr_only", "bcrl_only", "bcrh_only"],
    custom_splits=custom_splits,
    save_splits=False,
    cdr3l_col="IR_VJ_1_junction_aa",
    cdr3h_col="IR_VDJ_1_junction_aa",
    save_dir="models/IBD-Boland/classification/celltype",
)
```

## Inspect and save the results

`results` is keyed by modality. Each split's metrics contain accuracy, macro F1,
and weighted F1. The returned AnnData contains the subtype predictions and
embeddings.

```python
metrics = build_results_dataframe(results)
print(metrics)

output_dir = Path("outputs/IBD-Boland/classification/celltype")
output_dir.mkdir(parents=True, exist_ok=True)
metrics.to_csv(output_dir / "celltype_results.csv", index=False)
adata_classified.write_h5ad(output_dir / "adata_with_celltype_predictions.h5ad")
```

For [zero-shot RNA evaluation](03_BCR2RNA.md), reload the original processed
input. The classification output has been aligned to the checkpoint's genes
and should not be used as the measured RNA reference.
