# BCRFoundation

[![Documentation](https://img.shields.io/badge/docs-tutorials-157d85)](https://haotian-zhuang.github.io/BCRFoundation/)
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

The [pretrained checkpoint](pretrain_model/foundation_model_best.pt) is included
for downstream analyses.

## Tutorials

Detailed tutorials are available at this [website](https://haotian-zhuang.github.io/BCRFoundation/).

1. [Pretraining](https://haotian-zhuang.github.io/BCRFoundation/tutorials/01_pretraining.html)
2. [Cell-type classification](https://haotian-zhuang.github.io/BCRFoundation/tutorials/02_classification.html)
3. [Zero-shot BCR-to-RNA prediction](https://haotian-zhuang.github.io/BCRFoundation/tutorials/03_BCR2RNA.html)

See [data and checkpoint requirements](https://haotian-zhuang.github.io/BCRFoundation/data_and_checkpoints.html) before running
the tutorials. Each example uses the settings from the corresponding analysis.

## Citation

Citation information will be added with the corresponding publication.

## Contact

Author: Haotian Zhuang

Report bugs and provide suggestions by sending email to:

Maintainer: Haotian Zhuang (haotian.zhuang@duke.edu)

Or open a new issue on this GitHub page.
