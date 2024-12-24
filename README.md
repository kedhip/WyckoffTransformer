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
python tokenise_a_dataset.py mp_20 yamls/tokenisers/mp_20_sg_multiplicity.yaml --new-tokenizer
python train.py yamls/models/NextToken/v6/base_sg.yaml mp_20 cuda --pilot
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

Alternatively, you can use the cached tokeniser. This is important when a model that was trained on one dataset is  applied to a different dataset.
```bash
python tokenise_a_dataset.py <dataset-name> <path-to-tokenizer-yaml> --tokenizer-path cache/<dataset-names>/tokenisers/<tokenizer-name>.pkl.gz
```
# Training
```bash
python train.py <path-to-model-yaml> <dataset-name> <device>
```
The model weights are saved to `runs/<run-id>`, and to WanDB, along with the tokenizer.
1. If running on CRP, add `--torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}`
2. Adding `--pilot` will run the model for a small number of epochs.
3. Model configs contain the tokenizer name.

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

# matbench-discovery
## Data
As orginally desgined, models in matbench-discovery train on MPTrj and predict WBM.
We experimented with different training datasets:
1. `mp_2022` is MP 2022 – relaxed structures from Materials Project. Downloaded by this [notebook](scripts/data_preprocesssing/mp_2022.ipynb).
2. `mp_trj_full` is MPTrj – the full dataset, including both relaxed and unrelaxed structures. Downloaded by this  [notbook](scripts/data_preprocesssing/mptrj_extract_all.ipynb). Note that
symmetry changes only slighly during relaxation, meaning that after preprocessing the data a large number of
structures with the same Wyckoff representation have the same energy; [analysis](research_notebooks/mptrj_duplicates.ipynb).

If you modify the training data, be extremely careful that the target is _formation energy per atom_ and it's computed with same reference energies as WBM. Train / val split is entirely our choice, and can be modified freely.

The different symlinks in `data` allow to define variants of the datasets to be processed with different tolerances. The tolerance is set in the `cache_a_dataset.py` script and _is not done automatically_.

Tolerance didn't (2024) have a significant impact. Hence, for further experiments, just `mp_2022` seems to be a reasonable choice.
## Models
[This WanDB workspace](https://wandb.ai/symmetry-advantage/WyckoffTransformer?nw=wrbkiq2xgjk) is a good place to find some models and their performance.

# WanDB sweeps
TODO (Nikita)