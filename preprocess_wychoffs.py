from collections import defaultdict, Counter
import string
from pathlib import Path
import pickle
import gzip
from pyxtal import Group

from wyckoff_transformer.tokenization import FeatureEngineer

N_3D_SPACEGROUPS = 230


def enumerate_wychoffs_by_ss(output_file: Path = Path("cache", "wychoffs_enumerated_by_ss.pkl.gz")):
    """
    Enumerates all Wyckoff positions by site symmetry.

    Args:
        output_file (Path, optional): The output file. Defaults to "wychoffs_enumerated_by_ss.pkl.gz".
    """
    enum_from_ss_letter = defaultdict(dict)
    ss_from_letter = defaultdict(dict)
    letter_from_ss_enum = defaultdict(lambda: defaultdict(dict))
    multiplicity_from_ss_enum = defaultdict(lambda: defaultdict(dict))
    max_multiplicity = 0
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        group = Group(spacegroup_number)
        ss_counts = Counter()
        # [::-1] doesn't matter in principle,
        # but serves a cosmetic purpose, so that
        # a comes before b, etc.
        for wp in group.Wyckoff_positions[::-1]:
            wp.get_site_symmetry()
            ss_from_letter[spacegroup_number][wp.letter] = wp.site_symm
            enum_from_ss_letter[spacegroup_number][wp.letter] = ss_counts[wp.site_symm]
            letter_from_ss_enum[spacegroup_number][wp.site_symm][ss_counts[wp.site_symm]] = wp.letter
            multiplicity_from_ss_enum[spacegroup_number][wp.site_symm][ss_counts[wp.site_symm]] = wp.multiplicity
            max_multiplicity = max(max_multiplicity, wp.multiplicity)
            ss_counts[wp.site_symm] += 1
    with open(output_file, "wb") as f:
        pickle.dump((dict(enum_from_ss_letter), dict(letter_from_ss_enum),
            dict(ss_from_letter)), f)
    engineered_path = Path("cache", "engeneers")
    engineered_path.mkdir(exist_ok=True)
    multiplicity_engineer = FeatureEngineer(
        multiplicity_from_ss_enum, ("spacegroup_number", "site_symmetries", "sites_enumeration"),
        stop_token=max_multiplicity + 1, mask_token=max_multiplicity + 2, pad_token=0)
    with gzip.open(engineered_path / "multiplicity.pkl.gz", "wb") as f:
        pickle.dump(multiplicity_engineer, f)
    

def get_augmentation_dict():
    ascii_range = tuple(string.ascii_letters)
    alternatives_by_sg = {}
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        alternatives_letters = tuple(tuple(x.split()) for x in Group(spacegroup_number).get_alternatives()['Transformed WP'])
        reference_order = alternatives_letters[0]
        assert reference_order == ascii_range[:len(reference_order)]
        # There are transformations that don't rearrange Wychoff letters, e. g.
        # Group(1) has
        # {'No.': ['1', '2'],
        # 'Coset Representative': ['x,y,z', '-x,-y,-z'],
        # 'Geometrical Interpretation': ['1', '-1 0,0,0'],
        # 'Transformed WP': ['a', 'a']}
        alternatives_letters_set = frozenset(alternatives_letters)
        alternatives_this_sg = []
        for this_alternative in alternatives_letters_set:
            this_augmentator = dict()
            for new_letter, old_letter in zip(this_alternative, reference_order):
                this_augmentator[old_letter] = new_letter
            alternatives_this_sg.append(this_augmentator)
        alternatives_by_sg[spacegroup_number] = alternatives_this_sg
    return alternatives_by_sg


if __name__ == "__main__":
    enumerate_wychoffs_by_ss()
    print("Done enumerating Wyckoff positions inside site symmetry.")
    get_augmentation_dict()
    print("Done test-running Wyckoff positions augmentation.")