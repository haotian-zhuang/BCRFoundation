# Data and checkpoints

The tutorials use processed AnnData files (`.h5ad`). Their example paths are
relative to the repository root; replace them with your own input paths.
Research datasets are not included in the software repository.

## Gene expression and BCR sequences

Each row represents a cell. Gene identifiers belong in `adata.var_names`, and
cell identifiers belong in `adata.obs_names`.

| AnnData field | Contents |
| --- | --- |
| `adata.X` | Processed gene expression, with cells as rows and genes as columns |
| `adata.var_names` | Unique gene identifiers matching the identifier convention used for pretraining |
| `adata.obs["IR_VJ_1_junction_aa"]` | Light-chain CDR3 amino-acid sequences |
| `adata.obs["IR_VDJ_1_junction_aa"]` | Heavy-chain CDR3 amino-acid sequences |

Gene expression profiles were library-size-normalized and log-transformed,
followed by selection of highly variable genes. The pretraining tutorial starts
from normalized, log-transformed expression. It uses an existing
`adata.var["highly_variable"]` annotation, or selects 3,000 highly variable genes
with Scanpy's `seurat` flavor when that annotation is absent.

Use the same expression preprocessing as the checkpoint for downstream training
and evaluation. The package training functions consume the supplied expression
matrix; they do not perform normalization or log transformation for you.

The tutorial column names follow the analyses. If your BCR columns have different
names, pass those names through `cdr3l_col` and `cdr3h_col` consistently.

## B cell characteristics prediction inputs

The [characteristics prediction tutorial](tutorials/02_classification.md) uses
the following observation columns:

| Observation column | Contents |
| --- | --- |
| `tissue_name` | Tissue origin: `PBMC`, `Ileum`, or `Rectum` |
| `disease_name` | Disease status labels |
| `celltype_name` | B cell subtype labels |
| `receptor_subtype` | Light chain locus type labels |
| `split_by_sample` | Existing sample-based assignments: `train`, `val`, or `test` |

For tissue prediction, the tutorial merges `Ileum` and `Rectum` into `Intestine`
and retains the original labels in `tissue_name_original`. The other targets use
their existing labels. Only the target columns for the tasks you run are required.

The classifier excludes cells with a missing label for the selected task. Its
custom split indices must therefore refer to the remaining labeled cells in
their original order. The tutorial constructs these indices separately for each
task and preserves the supplied split annotations with `save_splits=False`.

## Zero-shot inputs and evaluation

Zero-shot prediction uses paired light- and heavy-chain CDR3 sequences. Expression
is not an input to the prediction model. With `evaluate=False`, `adata.X` is
optional.

The [zero-shot tutorial](tutorials/03_BCR2RNA.md) reproduces the analysis with
`evaluate=True`: measured RNA is used as ground truth, and all cells are evaluated
together without using split annotations. Use unique cell and gene identifiers.
Reload the original processed data for evaluation, rather than the gene-aligned
object returned by classification.

Predictions cover the checkpoint's gene set. Evaluation uses only measured genes
that overlap that set. Prediction values remain on the pretraining expression
scale; they are not converted back to raw counts.

## Pretrained checkpoint

The pretrained checkpoint is included in this repository. After cloning, the
downstream tutorials load it from:

```text
pretrain_model/foundation_model_best.pt
```

You can also [download the checkpoint directly](https://raw.githubusercontent.com/haotian-zhuang/BCRFoundation/main/pretrain_model/foundation_model_best.pt)
and save it at the same path. Run the tutorials from the repository root, or
adjust `checkpoint_path` to its location on your computer.

Alternatively, follow the [pretraining tutorial](tutorials/01_pretraining.md) to
train a model. That example writes its best checkpoint to:

```text
models/pretraining/foundation_model_best.pt
```

Set `checkpoint_path` to that file when using the newly trained model downstream.

Zero-shot prediction requires a checkpoint containing the dedicated
`bcr2gene_decoder_state_dict`, foundation-model weights, BCR vocabulary, and the
pretraining `gene_order`. Legacy checkpoints containing only the fused RNA
decoder are rejected by the zero-shot loader.
