from typing import List
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf

from evaluation.generated_dataset import GeneratedDataset, DATA_KEYS

def compute_fields_and_cache(data: GeneratedDataset) -> None:
    if "site_symmetries" not in data.data.columns:
        data.compute_wyckoffs()
    data.compute_wyckoff_fingerprints()
    if "numIons" not in data.data.columns:
        data.convert_wyckoffs_to_pyxtal()
    if "structure" in data.data.columns:
        data.compute_cdvae_crystals()
        data.compute_naive_validity()
    data.dump_to_cache()

def dive_and_cache(
    this_config: OmegaConf,
    transformations: List[str],
    dataset_name: str,
    config_file: Path,
    skip_cached: bool):

    if skip_cached:
        raise NotImplementedError("skip_cached is not implemented")
    if DATA_KEYS.intersection(this_config.keys()):
        print("Loading: ", transformations)
        data = GeneratedDataset.from_transformations(
            transformations, dataset=dataset_name)
        compute_fields_and_cache(data)

    for new_transformation, new_config in this_config.items():
        if new_transformation in DATA_KEYS:
            continue
        dive_and_cache(new_config, transformations + [new_transformation], dataset_name, config_file, skip_cached)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config-file", type=Path, default=Path(__file__).parent / "generated" / "datasets.yaml")
    parser.add_argument("--skip-cached", action="store_true")
    parser.add_argument("--dataset", type=str, nargs="+")
    args = parser.parse_args()
    if args.dataset:
        data = GeneratedDataset.from_transformations(args.dataset, config_path=args.config_file)
        compute_fields_and_cache(data)
    else:
        config = OmegaConf.load(args.config_file)
        for dataset_name, dataset_config in config.items():
            dive_and_cache(dataset_config, [], dataset_name, args.config_file, args.skip_cached)
           

if __name__ == "__main__":
    main()
