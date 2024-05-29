# Installation
1. Clone the repository
2. `poetry install`. If it doesn't work due to PyTorch, it's up to you to fix. If you are able to find a general solution, a PR is most welcome.
3. Log into WanDB, and configure your entity. For the team memebers, I suggest `symmetry-advantage`. It can be configured in poetry:
```bash
poetry self add poetry-dotenv-plugin
echo "WANDB_ENTITY=symmetry-advantage" > .env
```
# Training
`python train.py cuda:0|cuda:1|cpu`
# Generation
[Generate.ipynb](Generate.ipynb)
# Evaluation
[statistical_evaluation.ipynb](statistical_evaluation.ipynb)
# Developing
[Taskboard](https://www.notion.so/kna/36e263a83cc441a38483c084a5457a59?v=ecbd33a6130246bf940876abbf1d984c)
Please don't `main` branch
# Data sources
1. `mp_20` is [MP 20](https://github.com/txie-93/cdvae/tree/main/data/mp_20). To download, checkout the `cdvae` submodule.
2. `mp_20_biternary` is a selection of binary and ternary compounds from `mp_20`, produced with with `select_from_mp_20.py`.