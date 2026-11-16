# Installation

BCRFoundation is installed from its GitHub repository.

## Requirements

Use **Python 3.11.15 or later**. The minimum runtime dependency versions match
the environment used for the analyses:

| Package | Minimum version |
| --- | --- |
| PyTorch | 2.11.0 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| pandas | 2.3.3 |
| Scanpy | 1.11.5 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| tqdm | 4.67.3 |
| AnnData | 0.12.10 |

The tutorials select a CUDA device when available and otherwise use the CPU.

## Install from GitHub

Run these commands in the Python environment you will use for analysis:

```bash
git clone https://github.com/haotian-zhuang/BCRFoundation.git
cd BCRFoundation
python -m pip install -e .
```

The installer reads the runtime dependencies from `pyproject.toml`. Editable
installation makes changes in this checkout available to Python immediately.

## Verify installation

```bash
python -c "import BCRFoundation as bcrf; print(bcrf.__version__)"
```

Then prepare your [data and checkpoint](data_and_checkpoints.md), and choose a
tutorial:

- [Pretraining](tutorials/01_pretraining.md)
- [Cell-type classification](tutorials/02_classification.md)
- [Zero-shot BCR-to-RNA prediction](tutorials/03_BCR2RNA.md)
