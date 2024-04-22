# Welcome to DISCUS!

##  Overview

The Geometric Deep Learning for Diffusion MRI Signal Reconstruction with Continuous Samplings (DISCUS) method facilitates the flexible signal reconstruction for arbitrary q-vectors (query vectors) given an acquisition with an arbitrary number of measurements (observation set). 

![](/images/architecture.jpeg)


### Usage

This repository includes scripts to generate a dataset for training (`dataset.py`), to train the DISCUS method (`train.py`), and to predict diffusion MRI signals given a trained DISCUS model (`prediction.py`).
A few parameters and paths have to be set in the `config.yaml` file. This file needs to be referenced with the respective function call, e.g.
```
python train.py -f config.yaml
```

<!-- start of references -->
## Reference

If you use DISCUS for your research publication, please cite:

_Christian Ewert*, David Kügler*, Rüdiger Stirnberg, Alexandra Koch, Anastasia Yendiki, Martin Reuter (*co-first); Geometric Deep Learning for Diffusion MRI Signal Reconstruction with Continuous Samplings (DISCUS). Imaging Neuroscience 2024; 2 1–18. doi: https://doi.org/10.1162/imag_a_00121_
