import pathlib
import pymatgen.io.cif
from pymatgen.core import Structure
import pyxtal


def try_pyxtal(structure: Structure, tol: float) -> None:
    pyxtal_structure = pyxtal.pyxtal()
    try:
        pyxtal_structure.from_seed(seed=structure, tol=tol)
        print(f"Success with tol={tol}")
    except Exception as e:
        print(f"Failed with tol={tol}")
        print(e)


def main():
    structures =  pymatgen.io.cif.CifParser("no_can.cif").parse_structures()
    #for cif_name in pathlib.Path("no_can").glob("*.cif"):
    #print(f"Reading {cif_name}")
    #structure = pymatgen.io.cif.CifParser(cif_name).parse_structures()[0]
    for structure in structures:
        print(structure.composition)
        for tol in [0.1, 0.01, 0.001]:
            try_pyxtal(structure, tol)


if __name__ == "__main__":
    main()