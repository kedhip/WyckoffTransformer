from typing import Tuple
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure
from evaluation.generated_dataset import GeneratedDataset, load_all_from_config
from evaluation.novelty import NoveltyFilter, filter_by_unique_structure

def to_cif(structure: Structure) -> str:
    """
    Convert a pymatgen structure to a CIF string containing the primitive structure.
    """
    cif_writer = CifWriter(structure.to_primitive())
    return cif_writer.__str__()


def write_novel_structures(
    transformations: Tuple[str, ...],
    dataset: pd.DataFrame,
    save_path_root: Path,
    novel_save_count: int,
    novelty_filter: NoveltyFilter,
    truncation_heuristic: int = 3) -> None:

    dataset_head = dataset.iloc[:novel_save_count*truncation_heuristic]
    unique = filter_by_unique_structure(dataset_head)
    novel = novelty_filter.get_novel(unique)
    if len(novel) < novel_save_count:
        raise ValueError(f"Only {len(novel)} novel structures found, expected {novel_save_count}."
                          "Try increasing the dataset size or increasing truncation_heuristic in case"
                          "novelty is low.")
    novel_cif = novel.structure.iloc[:novel_save_count].apply(to_cif)
    site_counts = novel.structure.iloc[:novel_save_count].apply(lambda s: len(s.to_primitive().sites))
    print(f"Novel_{novel_save_count} primitive site counts: "
          f"{site_counts.mean():.1f} ± {site_counts.std():.1f}; max: {site_counts.max():.1f}")
    novel_cif.name = "cif"
    cif_path = save_path_root.joinpath(*transformations)
    cif_path.mkdir(parents=True, exist_ok=True)
    novel_cif.to_csv(cif_path.joinpath("cif.csv.gz"), index_label="index")    
    print(f"Saved {len(novel_cif)} novel CIFs to {cif_path}")


def main():
    mp_20_transofrmations = [
        ("WyckoffTransformer", "CrySPR", "CHGNet_fix"),
        ("WyckoffTransformer", "DiffCSP++"),
        ("CrystalFormer",),
        ("DiffCSP++",),
        ("DiffCSP", ),
        ("FlowMM", ),
    ]
    all_datasets = load_all_from_config(datasets=mp_20_transofrmations, dataset_name="mp_20")
    all_datasets[('WyCryst', 'CHGNet_fix')] = GeneratedDataset.from_cache(('WyCryst', 'CHGNet_fix'), "mp_20_biternary")
    novelty_reference = pd.concat([
        GeneratedDataset.from_cache(('split', 'train'), "mp_20").data,
        GeneratedDataset.from_cache(('split', 'val'), "mp_20").data], axis=0, verify_integrity=True)
    novelty_filter = NoveltyFilter(novelty_reference)

    novel_save_count = 105
    novel_save_path = Path(__file__).parent.joinpath("generated", "Dropbox", f"novel_{novel_save_count}")
    novel_save_path.mkdir(parents=True, exist_ok=True)

    for transformations in tqdm(mp_20_transofrmations):
        this_data = all_datasets[transformations].data
        write_novel_structures(transformations, this_data, novel_save_path, novel_save_count, novelty_filter)


if __name__ == "__main__":
    main()