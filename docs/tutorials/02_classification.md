# B cell characteristics prediction

Use BCRFoundation to predict tissue origin, disease status, cell subtype, and
light chain locus type. Provide the four label columns together to train
classifiers for all four tasks.

| Task | Label column in `adata.obs` |
| --- | --- |
| Tissue origin | `tissue_name` |
| Disease status | `disease_name` |
| Cell subtype | `celltype_name` |
| Light chain locus type | `receptor_subtype` |

First complete [installation](../installation.md) and review the
[data and checkpoint requirements](../data_and_checkpoints.md). Run the example
from the repository root and replace the input path with your data file.

## Load the data

The input should contain processed gene expression, paired CDR3 amino-acid
sequences, and ready-to-use labels for the tasks you want to predict. This
example also uses existing `train`, `val`, and `test` assignments in
`adata.obs["split_by_sample"]`.

```python
from pathlib import Path

import scanpy as sc
import BCRFoundation as bcrf
from BCRFoundation.finetune.utils import build_results_dataframe

adata = sc.read_h5ad("data/downstream/IBD-Boland/merged_processed_split.h5ad")
```

## Fine-tune the classifiers

Pass the label column names as a list. Each task compares RNA-only, paired-BCR,
individual BCR-chain, and combined RNA–BCR representations. The function handles
missing labels and saved split assignments separately for each task.

```python
results, adata_classified = bcrf.finetune.classification.train_classifier(
    adata,
    label_column=["tissue_name", "disease_name", "celltype_name", "receptor_subtype"],
    checkpoint_path="pretrain_model/foundation_model_best.pt",
    num_epochs=50,
    batch_size=512,
    modalities=["rna_only", "bcr_only", "bcrl_only", "bcrh_only", "rna_bcr"],
    split_column="split_by_sample",
    save_splits=False,
    cdr3l_col="IR_VJ_1_junction_aa",
    cdr3h_col="IR_VDJ_1_junction_aa",
    save_dir="models/IBD-Boland/classification",
)
```

Keep only the labels you need to run fewer tasks. Models are saved in a separate
subdirectory for each label column. The classifier with the best validation
accuracy is selected for each task and modality.

## Inspect and save the results

`results` is keyed by label column, then by modality. Each split's metrics contain
accuracy, macro F1, and weighted F1. For example, inspect cell subtype prediction:

```python
cell_subtype_metrics = build_results_dataframe(results["celltype_name"])
print(cell_subtype_metrics)
```

`adata_classified` contains the embeddings and predictions for all selected
tasks. Save them together:

```python
output_dir = Path("outputs/IBD-Boland/classification")
output_dir.mkdir(parents=True, exist_ok=True)
adata_classified.write_h5ad(output_dir / "adata_with_characteristic_predictions.h5ad")
```

For [zero-shot RNA evaluation](03_BCR2RNA.md), reload the original processed
input. The classification output has been aligned to the checkpoint's genes
and should not be used as the measured RNA reference.
