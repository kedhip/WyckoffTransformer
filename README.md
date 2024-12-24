# Installation
1. Clone the repository
3. Copy `pyproject.toml.CRP` to `pyproject.toml`, edit it as nesessary for your environemnt, run `poetry install`.
4. Log into WanDB, and configure your entity. For the team members, I suggest `symmetry-advantage`. It can be configured in poetry:
```bash
poetry self add poetry-dotenv-plugin
echo "WANDB_ENTITY=symmetry-advantage" > .env
```
5. Preprocess the data on Wychoff positions:
```bash
python preprocess_wychoffs.py
```
# Running a pilot model
Next token prediction:
```bash
python cache_a_dataset.py mp_20
python tokenise_a_dataset.py mp_20 TODO
```
# Training Data Preprocessing
The available datasets correspond to the folders in `data` and `cdvae/data`. Dataset idetifiers are the folder names, they are used throught the project. Note that some of the folders are symlinks. For data to be used for training, we need to do two preprocessing steps:
## Compute and cache symmetry information
```bash
python cache_a_dataset.py <dataset-name>
```
This will create a pickled representaiton of the dataset in `cache/<dataset-name>/data.pkl.gz`. The script supports setting symmetry tolerance, _this is not done automatically_, despite the names of some of the datasets.
## Tokenization
The tokenization script serves two purposes: it produces the mapping from the real data to token ids, and saves
tensor. To produce a new tokenizer:
```bash
python tokenise_a_dataset.py <dataset-name> <path-to-tokenizer-yaml> --new-tokenizer
```
Tokenizer configs are stored in `yamls/tokenisers`. The tokeniser is saved to `cache/<dataset-names>/tokenisers/**.pkl.gz`, preserving the folder structure of the config.

Alternatively, you can use the cached tokeniser. This is important is a model that was trained on a dataset is to be applied to a different dataset.
```bash
python tokenise_a_dataset.py <dataset-name> <path-to-tokenizer-yaml> --tokenizer-path ache/<dataset-names>/tokenisers/<tokenizer-name>.pkl.gz
```
# Training
```bash
python train.py <path-to-model-yaml> <dataset-name> <device>
```
1. If running on CRP, add --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}
2. Adding `--pilot` will run the model for a small number of epochs.
The model weights are saved to `runs/<run-id>`, and to WanDB, along with the tokenizer.

# Experiments
## Models - next token
1. Used for ILCR submission: `yamls/models/NextToken/v6/base_sg.yaml`
2. Latest (2024) harmonic, predicting enumerations: `yamls/models/mp_20/NextToken/augmented_harmonic_no_dropout.yaml`
3. Latest (2024) harmonic, predicting harmonic clusters (generation of actual structures WIP): `yamls/models/mp_20/NextToken/augmented/cluster_harmonic.yaml`
## Models - properties
### MP-20
Found by a hyperparameter search aka WanDB sweep:
1. `yamls/models/mp_20/band_gap/iclr_2025.yaml`
2. `yamls/models/mp_20/formation_energy_per_site/iclr_2025.yaml`
### AFLOW & 3DSC
TODO (Ignat)

## Get the cached preprocessed data and model weights
TODO
## Energy and band gap prediction
```bash
python train.py yamls/models/base_sg_energy.yaml mp_20 cuda:0 --pilot
```

# ICLR 2025 model training
## Next token
```bash
python train.py 
```
## Energy and band gap prediction
```bash
python train.py yamls/models/base_sg_energy.yaml mp_20 cuda:0 
python train.py yamls/models/base_sg_band_gap.yaml mp_20 cuda:0 
```

# ICLR 2025 data preprocessing
If you just want to reproduce the ML parts, you should
download the cached data from the cloud. Redoing tokenization
is not deterministic.
```bash
python cache_a_dataset.py mp_20
python cache_a_dataset.py wbm
python tokenise_a_dataset.py mp_20 yamls/tokenisers/mp_20_sg_multiplicity_scalars.yaml --new-tokenizer
```

# WanDB sweeps
TODO (Nikita)

# matbench-discovery
A big WIP.