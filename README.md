# Installation
1. Clone the repository
2. `poetry install`. If it doesn't work due to PyTorch, it's up to you to fix. If you are able to find a general solution, a PR is most welcome. We don't really need this specific git tag, anything `2.3` and `2.4` should work.
3. Log into WanDB, and configure your entity. For the team memebers, I suggest `symmetry-advantage`. It can be configured in poetry:
```bash
poetry self add poetry-dotenv-plugin
echo "WANDB_ENTITY=symmetry-advantage" > .env
```
# Data preprocessing
```bash
# Preprocess the data on Wychoff positions
python preprocess_wychoffs.py
# Read CIF structures and find the Wychoff positions
# A CPU-intensive operations
python cache_a_dataset.py mp_20
# Tokenise a dataset
python tokenise_a_dataset.py mp_20 yamls/tokenisers/mp_20_CSP.yaml
```
# Training
To test training on CPU and GPU:
```bash
# If running on CRP, add --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}
python train.py yamls/models/NextToken/v6/pilot.yaml mp_20 cpu
python train.py yamls/models/NextToken/v6/pilot.yaml mp_20 cuda:0
```
# Developing
[Taskboard](https://www.notion.so/kna/36e263a83cc441a38483c084a5457a59?v=ecbd33a6130246bf940876abbf1d984c)
Please don't `main` branch
# Data sources
1. `mp_20` is [MP 20](https://github.com/txie-93/cdvae/tree/main/data/mp_20). To download, checkout the `cdvae` submodule.
2. `mp_20_biternary` is a selection of binary and ternary compounds from `mp_20`, produced with with `select_from_mp_20.py`.