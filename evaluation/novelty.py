from typing import Dict
from pandas import Series
def record_to_augmented_fingerprint(row: Dict|Series) -> tuple:
    """
    Computes a fingerprint taking into account equivalent Wyckoff position enumeration.
    Args:
        row contains the Wyckoff information:
        - spacegroup_number
        - elements
        - site_symmetries
        - sites_enumeration_augmented
    Returns:
        A tuple of the spacegroup number and a frozenset of tuples of the elements,
        site symmetries and the Wyckoff position enumeration.
    """
    transposed_augmentations = zip(*row["sites_enumeration_augmented"])
    return (
        row["spacegroup_number"],
        frozenset(
            map(
                tuple,
                zip(row["elements"], row["site_symmetries"], *transposed_augmentations)
            )
        )
    )