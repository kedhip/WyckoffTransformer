# Installation
1. clone the repository
2. `poetry install`. If it doesn't work due to torch, it's up to you to fix. If you are able to find a general solution, a PR is most welcome.
# Training
1. Log into WanDB, and configure your entity. For the team memebers, I suggest `export WANDB_ENTITY="symmetry-advantage"`
2. `python train.py cuda:0|cuda:1|cpu`
# Generation
[Generate.ipynb](Generate.ipynb)
# Evaluation
[statistical_evaluation.ipynb](statistical_evaluation.ipynb)
# Developing
[Taskboard](https://www.notion.so/kna/36e263a83cc441a38483c084a5457a59?v=ecbd33a6130246bf940876abbf1d984c)
# Data sources
1. `mp_20` is [MP 20](https://github.com/txie-93/cdvae/tree/main/data/mp_20). To download, checkout the `cdvae` submodule.
2. `mp_20_biternary` is a selection of binary and ternary compounds from `mp_20`, produced with with `select_from_mp_20.py`.