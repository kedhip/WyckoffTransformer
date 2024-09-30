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
        frozenset of all possible Wyckoff representations of the structure.
    """
    return (
        row["spacegroup_number"],
        frozenset(            
            map(lambda enumertaion:
                frozenset(
                    map(
                        tuple,
                        zip(row["elements"], row["site_symmetries"], enumertaion)
                    )
                ), row["sites_enumeration_augmented"]
            )
        )
    )