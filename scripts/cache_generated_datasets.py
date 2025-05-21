if __name__ == "__main__":
    # We want to avoid messing with the environment variables in case we are used as a module.
    # The code is parallelised by structure
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    # Add the project root to sys.path to allow imports from the root directory
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Optional
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf

from evaluation.generated_dataset import GeneratedDataset, DATA_KEYS

def compute_fields_and_cache(data: GeneratedDataset) -> GeneratedDataset:
    if "site_symmetries" not in data.data.columns:
        data.compute_wyckoffs()
    data.compute_wyckoff_fingerprints()
    if "numIons" not in data.data.columns:
        data.convert_wyckoffs_to_pyxtal()
    data.compute_smact_validity()
    if "structure" in data.data.columns:
        data.compute_cdvae_crystals()
        data.compute_naive_validity()
        try:
            import torch_scatter
            import torch_sparse
            data.compute_cdvae_e()
        except ImportError as e:
            print("Required libraries are not installed. Skipping cdvae_e computation.")
            print("Error message:", e)

    data.dump_to_cache()
    return data

def dive_and_cache(
    this_config: OmegaConf,
    transformations: List[str],
    dataset_name: str,
    config_file: Path,
    last_transformation: Optional[str] = None) -> None:

    if DATA_KEYS.intersection(this_config.keys()) and ( # type: ignore
        not last_transformation or last_transformation == transformations[-1]):
        
        print(f"From {dataset_name} loading ({', '.join(transformations)})")
        data = GeneratedDataset.from_transformations(
            transformations, dataset=dataset_name)
        compute_fields_and_cache(data)

    for new_transformation, new_config in this_config.items():
        if new_transformation in DATA_KEYS:
            continue
        dive_and_cache(new_config, transformations + [new_transformation], dataset_name, config_file, last_transformation)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config-file", type=Path, default=Path(__file__).parent.parent / "generated" / "datasets.yaml") # Adjusted path
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--transformations", type=str, nargs="+")
    parser.add_argument("--last-transformation", type=str,
                        help="Only process the datasets with this transformation as the last one")
    args = parser.parse_args()
    if args.transformations:
        data = GeneratedDataset.from_transformations(
            args.transformations, config_path=args.config_file, dataset=args.dataset)
        if len(data.data) == 0:
            raise ValueError("The dataset has zero length")
        compute_fields_and_cache(data)
    else:
        config = OmegaConf.load(args.config_file)
        for dataset_name, dataset_config in config.items(): # type: ignore
            if args.dataset and args.dataset != dataset_name:
                continue
            dive_and_cache(dataset_config, [], str(dataset_name), args.config_file, args.last_transformation) # Added str() cast


if __name__ == "__main__":
    main()
