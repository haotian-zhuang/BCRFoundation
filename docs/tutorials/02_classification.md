# B cell characteristics prediction

Fine-tune BCRFoundation to predict four B cell characteristics using the settings
from the IBD-Boland analyses. Each task compares RNA-only, paired-BCR, individual
BCR-chain, and combined RNA–BCR representations.

| Task | Label column in `adata.obs` | Output folder |
| --- | --- | --- |
| Tissue origin | `tissue_name` | `tissue` |
| Disease status | `disease_name` | `disease` |
| Cell subtype | `celltype_name` | `celltype` |
| Light chain locus type | `receptor_subtype` | `locus` |

First complete [installation](../installation.md) and prepare the inputs
listed in [data and checkpoints](../data_and_checkpoints.md). Run the code from
the repository root. The example expects a compatible checkpoint at
`pretrain_model/foundation_model_best.pt`; change that path if using a model
from [pretraining](01_pretraining.md).

## Prepare the inputs

The processed input must contain normalized, log-transformed expression in
`adata.X`, paired CDR3 amino-acid columns, the target labels listed above, and
saved sample-based split labels in `split_by_sample`.

For tissue origin, the analysis combines **Ileum and Rectum into Intestine**,
producing a PBMC-versus-Intestine task. The original anatomical labels are
retained in `tissue_name_original`. The other three targets use their existing
labels without recoding.

```python
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import BCRFoundation as bcrf
from BCRFoundation.finetune.utils import build_results_dataframe

cdr3l_col = "IR_VJ_1_junction_aa"
cdr3h_col = "IR_VDJ_1_junction_aa"
split_column = "split_by_sample"
```

## Define one function for all four tasks

`train_classifier` accepts one label column at a time. The helper below accepts
all four columns together and trains a separate classifier for each task and
modality. Every task uses 50 epochs, a batch size of 512, and the same five
modalities as the analyses.

The helper reloads the original data and resets the random seeds to zero for
each task, matching the separate analysis scripts. It constructs split indices
**after excluding cells with a missing label for that task**. All three groups
must be represented, with no missing or unknown split values among labeled
cells. `save_splits=False` retains the existing split annotations.

```python
def predict_b_cell_characteristics(data_path, checkpoint_path, label_columns):
    results_by_task = {}

    for task_name, label_column in label_columns.items():
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        adata = sc.read_h5ad(data_path)
        required = {label_column, split_column, cdr3l_col, cdr3h_col}
        missing = required.difference(adata.obs.columns)
        if missing:
            raise KeyError(f"Missing adata.obs columns: {sorted(missing)}")

        if label_column == "tissue_name":
            original = adata.obs[label_column].astype("string")
            unexpected = set(original.dropna()) - {"PBMC", "Ileum", "Rectum"}
            if unexpected:
                raise ValueError(f"Unexpected tissue labels: {sorted(unexpected)}")
            adata.obs["tissue_name_original"] = original.astype("category")
            adata.obs[label_column] = original.replace(
                {"Ileum": "Intestine", "Rectum": "Intestine"}
            ).astype("category")

        split_values = adata.obs.loc[
            adata.obs[label_column].notna(), split_column
        ].astype("string")
        if split_values.isna().any() or set(split_values.unique()) != {
            "train", "val", "test"
        }:
            raise ValueError(
                f"{label_column}: labeled cells need nonmissing train/val/test splits"
            )
        split_array = split_values.to_numpy()
        custom_splits = tuple(
            np.flatnonzero(split_array == name) for name in ("train", "val", "test")
        )

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
            save_dir=str(Path("models/IBD-Boland/classification") / task_name),
        )

        output_dir = Path("outputs/IBD-Boland/classification") / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        build_results_dataframe(results).to_csv(
            output_dir / f"{task_name}_results.csv", index=False
        )
        adata_classified.write_h5ad(
            output_dir / f"adata_with_{task_name}_embeddings_predictions.h5ad"
        )
        results_by_task[task_name] = results
        del adata, adata_classified

    return results_by_task
```

The package aligns expression genes to the checkpoint's gene order and selects
the best classifier for each modality using validation accuracy. Each task
starts from the supplied pretrained checkpoint.

## Run the four tasks together

Pass all four label columns in one call. To run only selected tasks, keep just
those entries in `label_columns`.

```python
results_by_task = predict_b_cell_characteristics(
    data_path="data/downstream/IBD-Boland/merged_processed_split.h5ad",
    checkpoint_path="pretrain_model/foundation_model_best.pt",
    label_columns={
        "tissue": "tissue_name",
        "disease": "disease_name",
        "celltype": "celltype_name",
        "locus": "receptor_subtype",
    },
)
```

## Inspect the results

Each split's metrics are returned as accuracy, macro F1, and weighted F1.
The helper saves a separate CSV and AnnData file for each task under
`outputs/IBD-Boland/classification/<task>/`; the AnnData files contain that task's
embeddings and predictions. Fine-tuned models are saved under
`models/IBD-Boland/classification/<task>/`.

```python
for task_name, results in results_by_task.items():
    for modality, metrics in results.items():
        accuracy, f1_macro, f1_weighted = metrics["test"]
        print(
            f"{task_name} / {modality}: accuracy={accuracy:.3f}, "
            f"macro F1={f1_macro:.3f}, weighted F1={f1_weighted:.3f}"
        )
```

For [zero-shot RNA evaluation](03_BCR2RNA.md), reload the original processed
input as shown in that tutorial. Classification aligns and fills missing gene
columns, so its returned expression matrix should not be used as the measured
RNA reference for that evaluation.
