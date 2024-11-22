from collections import defaultdict, Counter
import string
from pathlib import Path
import pickle
import gzip
import numpy as np
from pyxtal import Group

from wyckoff_transformer.tokenization import FeatureEngineer

N_3D_SPACEGROUPS = 230


from scipy.special import sph_harm

def convolve_vectors_with_spherical_harmonics(vectors_batch, degree):
    """
    Convolves a batch of 3D vectors with spherical harmonics without explicit loops.
    
    Parameters:
    vectors_batch : ndarray
        A 3D array of shape (num_batches, num_objects, 3) representing the vectors.
    degree : int
        The degree of the spherical harmonics.
    order : int
        The order of the spherical harmonics.
    
    Returns:
    ndarray
        A 1D array of shape (num_batches,) with the convolved values for each batch.
    """
    # Normalize the vectors
    norms = np.linalg.norm(vectors_batch, axis=-1, keepdims=True)
    x, y = vectors_batch[..., 0], vectors_batch[..., 1]
    phi = np.arctan2(y, x)
    # Avoid division by zero
    nonzero_norms = norms.copy()
    nonzero_norms[nonzero_norms == 0] = 1
    z = vectors_batch[..., 2] / nonzero_norms.squeeze(-1)
    theta = np.arccos(z)

    # Compute spherical harmonics for all vectors
    res = np.array([sph_harm(order, degree, phi, theta) for order in range(degree+1)])
    res *= np.expand_dims(norms.squeeze(-1), 0)
    return res.mean(axis=-1)

def enumerate_wychoffs_by_ss(output_file: Path = Path("cache", "wychoffs_enumerated_by_ss.pkl.gz")):
    """
    Enumerates all Wyckoff positions by site symmetry.

    Args:
        output_file (Path, optional): The output file. Defaults to "wychoffs_enumerated_by_ss.pkl.gz".
    """
    enum_from_ss_letter = defaultdict(dict)
    translation_sg_ss_enum = defaultdict(dict)
    ss_from_letter = defaultdict(dict)
    letter_from_ss_enum = defaultdict(lambda: defaultdict(dict))
    multiplicity_from_ss_enum = dict()
    max_multiplicity = 0
    reference_vectors = (
        np.array([0, 0, 0]),
        np.array([1, 1, 1]),
    )
    ss_corrections = {
        150: {
            "f": ".2.",
            "e": ".2."},
        152: {
            "a": ".2.",
            "b": ".2."},
        154: {
            "a": ".2.",
            "b": ".2."},
        155: {
            "e": ".2",
            "d": ".2"},
        164: {
            "g": ".2.",
            "h": ".2."},
        165: {
            "f": ".2."},
        166: {
            "f": ".2",
            "g": ".2"},
        167: {
            "e": ".2"
        },
        177: {
            "j": ".2.",
            "k": ".2."},
        178: {
            "a": ".2."
        },
        179: {
            "a": ".2."
        },
        180: {
            "g": ".2.",
            "h": ".2."},
        181: {
            "g": ".2.",
            "h": ".2."},
        182: {
            "g": ".2."},
        190: {
            "g": ".2."},
        191: {
            "o":  ".m."},
        192: {
            "j": ".2."},
        194: {
            "i": ".2.",
            "k": ".m."}
    }
    for spacegroup_number in range(1, N_3D_SPACEGROUPS + 1):
        group = Group(spacegroup_number)
        ss_counts = Counter()
        opres_by_ss_enum = defaultdict(dict)
        # [::-1] doesn't matter in principle,
        # but serves a cosmetic purpose, so that
        # a comes before b, etc.
        for wp in group.Wyckoff_positions[::-1]:
            wp.get_site_symmetry()
            # https://github.com/MaterSim/PyXtal/issues/295
            try:
                site_symm = ss_corrections[spacegroup_number][wp.letter]
            except KeyError:
                site_symm = wp.site_symm
            ss_from_letter[spacegroup_number][wp.letter] = site_symm
            enum_from_ss_letter[spacegroup_number][wp.letter] = ss_counts[site_symm]
            opres_by_ss_enum[site_symm][ss_counts[site_symm]] = \
                [[op.operate(v) for op in wp] for v in reference_vectors]
            #if site_symm not in translation_base_by_ss:
            #    translation_base_by_ss[site_symm] = all_translations
            #else:
            #    if np.linalg.norm(translation_base_by_ss[site_symm]) > \
            #        np.linalg.norm(all_translations):
            #        print(f"Spacegroup {spacegroup_number}, Wyckoff position {wp.letter}")
            #        print("Translatio for the junior position:")
            #        print(translation_base_by_ss[site_symm])
            #        print("Translations:")
            #        print(all_translations)
            #        raise AssertionError("Translations are not the same for all operations.")
            #all_ops_translations = all_translations - translation_base_by_ss[site_symm]
            #if not (all_ops_translations == all_ops_translations[0]).all():
            #    print(f"Spacegroup {spacegroup_number}, Wyckoff position {wp.letter}")
            #    print("Translatio for the junior position:")
            #    print(translation_base_by_ss[site_symm])
            #    print("Translations:")
            #    print(all_ops_translations)
            #    print(all_translations)
            #    raise AssertionError("Translations are not the same for all operations.")
            #translation_from_sg_letter[spacegroup_number][wp.letter] = \
            #    np.concatenate((all_ops_translations.mean(axis=0), all_ops_translations.std(axis=0)))
            #print(all_translations.shape)
            letter_from_ss_enum[spacegroup_number][site_symm][ss_counts[site_symm]] = wp.letter
            multiplicity_from_ss_enum[(spacegroup_number, site_symm, ss_counts[site_symm])] = wp.multiplicity
            max_multiplicity = max(max_multiplicity, wp.multiplicity)
            ss_counts[site_symm] += 1
        #print(f"Spacegroup {spacegroup_number} done.")
        #print(translation_from_ss_letter[spacegroup_number])
        for ss, opres_by_enum in opres_by_ss_enum.items():
            print(f"Spacegroup {spacegroup_number}, wp {ss} {letter_from_ss_enum[spacegroup_number][ss]}")
            # Step 1: find the position closest to the origin
            all_ops = np.concatenate([np.expand_dims(a, 0) for a in opres_by_enum.values()], axis=0)
            # [enum][ref_vector][op][xyz]
            print("Ops [enum][ref_vector][op][xyz]:")
            print(all_ops.shape)
            degree = 6
            signatures = convolve_vectors_with_spherical_harmonics(all_ops, degree)#.reshape(len(all_ops), degree + 1)
            print("Signatures [degree][enum][ref_vector]")
            print(signatures.shape)
            signatures = signatures.reshape(degree + 1, len(opres_by_enum), len(reference_vectors))
            assert np.unique(signatures, axis=1).shape == signatures.shape
            ##assert np.abs(np.abs(norm_ops) - np.abs(norm_ops[0])).sum() < 1e-6
            #origin_proximity_per_op = np.linalg.norm(all_ops, axis=2)
            #print("Origin proximity per op:")
            #print(origin_proximity_per_op)
            #origin_proximity = origin_proximity_per_op.mean(axis=1)
            #reference_op_idx = np.argmin(origin_proximity)
            #print("Origin proximity:")
            #print(origin_proximity)
            #assert (origin_proximity - origin_proximity[reference_op_idx] < 1e-6).sum() == 1
            
            #refernce_op = all_ops[np.argmin(origin_proximity)]
            # Step 2: get the signatures
            #shifted_ops = all_ops - refernce_op
            
            
    with open(output_file, "wb") as f:
        pickle.dump((dict(enum_from_ss_letter), dict(letter_from_ss_enum),
            dict(ss_from_letter)), f)
    engineered_path = Path("cache", "engineers")
    engineered_path.mkdir(exist_ok=True)
    multiplicity_engineer = FeatureEngineer(
        multiplicity_from_ss_enum, ("spacegroup_number", "site_symmetries", "sites_enumeration"),
        name="multiplicity",
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