from collections import defaultdict, Counter
from pathlib import Path
import pickle
import json
from pyxtal import Group


N_3D_SPACEGROUPS = 230


def enumerate_wychoffs_by_ss(output_file: Path = "wychoffs_enumerated_by_ss.pkl.gz"):
    """
    Enumerates all Wyckoff positions by site symmetry.

    Args:
        output_file (Path, optional): The output file. Defaults to "wychoffs_enumerated_by_ss.pkl.gz".
    """
    enum_from_ss_letter = defaultdict(dict)
    ss_from_letter = defaultdict(dict)
    letter_from_ss_enum = defaultdict(lambda: defaultdict(dict))
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        group = Group(spacegroup_number)
        ss_counts = Counter()
        for wp in group.Wyckoff_positions:
            wp.get_site_symmetry()
            ss_from_letter[spacegroup_number][wp.letter] = wp.site_symm
            enum_from_ss_letter[spacegroup_number][wp.letter] = ss_counts[wp.site_symm]
            letter_from_ss_enum[spacegroup_number][wp.site_symm][ss_counts[wp.site_symm]] = wp.letter
            ss_counts[wp.site_symm] += 1
    with open(output_file, "wb") as f:
        pickle.dump((dict(enum_from_ss_letter), dict(letter_from_ss_enum), dict(ss_from_letter)), f)


def parse_wp_lists(wp_lists_str: str):
    wp_lists_str = wp_lists_str.strip('[]')
    lists_str = wp_lists_str.split(', ')
    lists = [s.split(' ') for s in lists_str]
    return lists


def parse_equivalent_wp_sets(
    input_json: Path = "wyckoff_sets.json",
    output_file: Path = "wyckoff_lists.pkl.gz"):
    with open(input_json, "r", encoding="ascii") as f:
        wyckoff_sets_raw = json.load(f)

    wychoff_lists = {}
    for spacegroup_number_str, wp_lists_str in wyckoff_sets_raw.items():
        wychoff_lists[int(spacegroup_number_str)] = parse_wp_lists(wp_lists_str)
    
    with open(output_file, "wb") as f:
        pickle.dump(wychoff_lists, f)


def prepare_augmentation_tensor(
    wychoff_lists: dict,
    site_to_ids: dict,
    spacegroup_to_ids: dict,
    enum_from_ss_letter: dict,
    ss_from_letter: dict,
    device):
    # Prepare the following tensors:
    # augmentation_index = torch.randint(0, augmentation_count[space_group])
    # new_enum = augmentator[space_group][augmentation_index][site_symmetry][old_enum]
    # with space_group and site_symmetry being ids, not the real-world values
    # in the future, it will be made into sparse tensors
    augmentator = []
    for spacegroup_id in range(len(spacegroup_to_ids)): # pylint: disable=consider-using-enumerate  
        spacegroup_number = spacegroup_to_ids[spacegroup_id]
        augmentator.append([])
        default_letter_order = wychoff_lists[spacegroup_number][0]
        for this_wychoff_list in wychoff_lists[spacegroup_number]:
            this_augmentator = dict()
            for new_letter, old_letter in zip(this_wychoff_list, default_letter_order):
                site_symmetry = ss_from_letter[spacegroup_number][new_letter]
                assert site_symmetry == ss_from_letter[spacegroup_number][old_letter]
                this_augmentator[site_symmetry][enum_from_ss_letter[spacegroup_number][old_letter]] = \
                    enum_from_ss_letter[spacegroup_number][new_letter]
            augmentator[-1].append(this_augmentator)


if __name__ == "__main__":
    parse_equivalent_wp_sets()
    enumerate_wychoffs_by_ss()
