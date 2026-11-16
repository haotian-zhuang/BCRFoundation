# BCRFoundation

[![Documentation](https://img.shields.io/badge/docs-tutorials-157d85)](docs/index.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11.15+](https://img.shields.io/badge/python-3.11.15+-blue.svg)](https://www.python.org/downloads/)

BCRFoundation is a multimodal foundation model pretrained on scPAIRB, a curated corpus containing over 0.6 million B cells with paired single-cell gene expression profiles and BCR sequences. BCRFoundation supports diverse downstream applications, including prediction of B cell characteristics and antigen specificity, as well as zero-shot prediction of gene expression profiles directly from BCR sequences.

![Figure 1. Overview of BCRFoundation.](figure/Figure1.png)

## Installation

```bash
git clone https://github.com/haotian-zhuang/BCRFoundation.git
cd BCRFoundation
pip install -e .
```

**Requirements**: Python 3.11.15+, PyTorch 2.11.0+, SciPy 1.17.1+

The [pretrained checkpoint](pretrain_model/foundation_model_best.pt) is included
for downstream analyses.

## Tutorials

Detailed examples are available in the [documentation](docs/index.md):

1. [Pretraining](docs/tutorials/01_pretraining.md)
2. [Cell-type classification](docs/tutorials/02_classification.md)
3. [Zero-shot BCR-to-RNA prediction](docs/tutorials/03_BCR2RNA.md)

See [data and checkpoint requirements](docs/data_and_checkpoints.md) before running
the tutorials. Each example uses the settings from the corresponding analysis.

## Citation

Citation information will be added with the corresponding publication.

## Contact

Author: Haotian Zhuang

Report bugs and provide suggestions by sending email to:

Maintainer: Haotian Zhuang (haotian.zhuang@duke.edu)

Or open a new issue on this GitHub page.
