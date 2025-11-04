from importlib.resources import files
from pathlib import Path
import pickle
from typing import Any, Optional, Sequence, Union
import warnings
from ase import Atoms
from ase.io import write
import lmdb
import pandas as pd
from pymatgen.core import Structure as PymatgenStructure, Lattice as PymatgenLattice
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Structure(object):
    """
    Storage for a single crystalline structure of atoms.

    The structure is represented by its cell, atom numbers, and real atom coordinates.  # TODO: This should probably be fractional.
    Additionally, one can store arbitrary properties and metadata associated with the structure in this class.

    :param cell:
        A 3x3 matrix of the lattice vectors. The [i, j]-th element is the jth Cartesian coordinate of the ith unit
        vector.
    :type cell: torch.Tensor
    :param atomic_numbers:
        A list of N integers giving the atomic numbers of the atoms, where N is the number of atoms.
    :type atomic_numbers: Sequence[int]
    :param pos:
        A Nx3 matrix of the real coordinates of the atoms in the structure, where N is the number of atoms.
    :type pos: torch.Tensor
    :param property_dict:
        An optional dictionary of properties associated with the structure.
        Defaults to None.
    :type property_dict: Optional[dict[str, Any]]
    :param metadata:
        An optional dictionary of metadata associated with the structure.
        Defaults to None.
    :type metadata: Optional[dict[str, Any]]

    :raises AssertionError:
        If the cell is not a 3x3 matrix or if the pos is not a Nx3 matrix, where N is the length of species.
    """

    def __init__(self, cell: torch.Tensor, atomic_numbers: Sequence[int], pos: torch.Tensor,
                 property_dict: Optional[dict[str, Any]] = None, metadata: Optional[dict[str, Any]] = None) -> None:
        """Constructor for the Structure class."""
        assert cell.shape == (3, 3)
        assert pos.shape == (len(atomic_numbers), 3)
        self._cell = cell
        self._atomic_numbers = list(atomic_numbers)
        self._pos = pos
        self._property_dict = property_dict if property_dict is not None else {}
        self._metadata = metadata if metadata is not None else {}

    @property
    def cell(self) -> torch.Tensor:
        """
        Return the cell vectors of the structure.

        :return:
            A 3x3 matrix of the lattice vectors. The [i, j]-th element is the jth Cartesian coordinate of the ith unit
            vector.
        :rtype: torch.Tensor
        """
        return self._cell

    @property
    def atomic_numbers(self) -> list[int]:
        """
        Return the atomic numbers of the atoms in the structure.

        :return:
            A list of N integers giving the atomic numbers of the atoms, where N is the number of atoms.
        :rtype: list[int]
        """
        return self._atomic_numbers

    @property
    def pos(self) -> torch.Tensor:
        """
        Return the real coordinates of the atoms in the structure.

        :return:
            A Nx3 matrix of the real coordinates of the atoms, where N is the number of of atoms.
        :rtype: torch.Tensor
        """
        return self._pos

    @property
    def property_dict(self) -> dict[str, Any]:
        """
        Return the property dictionary of the structure.

        :return:
            A dictionary of properties associated with the structure.
        :rtype: dict[str, Any]
        """
        return self._property_dict

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return the metadata of the structure.

        :return:
            A dictionary of metadata associated with the structure.
        :rtype: dict[str, Any]
        """
        return self._metadata

    def to(self, floating_point_precision: torch.dtype) -> None:
        """
        Convert the floating point precision of the structure to the given torch precision.

        :param floating_point_precision:
            The torch floating point dtype to convert to.
            Should be one of torch.float32, torch.float64, torch.float16, torch.bfloat16,
            torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz, torch.float8_e8m0fnu, or torch.float4_e2m1fn_x2.
        :type floating_point_precision: torch.dtype

        :raises ValueError:
            If the given floating point precision is not supported.
        """
        # See https://docs.pytorch.org/docs/stable/tensor_attributes.html.
        if not floating_point_precision in (torch.float32, torch.float64, torch.float16, torch.bfloat16,
                                            torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz,
                                            torch.float8_e5m2fnuz, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
            raise ValueError(f"Unsupported floating point precision: {floating_point_precision}. "
                             f"Supported precisions are torch.float32, torch.float64, torch.float16, "
                             f"torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, "
                             f"torch.float8_e4m3fnuz, torch.float8_e5m2fnuz, torch.float8_e8m0fnu, "
                             f"and torch.float4_e2m1fn_x2.")
        self._cell = self._cell.to(floating_point_precision)
        self._pos = self._pos.to(floating_point_precision)
        for key, value in self._property_dict.items():
            if torch.is_tensor(value) and value.is_floating_point():
                self._property_dict[key] = value.to(floating_point_precision)


class StructureDataset(Dataset):
    """
    Dataset for reading and storing a list of crystalline structures from several file formats.

    The supported file formats are LMDB, xyz, and csv. The specific format is inferred from the file extension.
    The _from_lmdb, _from_xyz, and _from_csv methods are used to read the structures from the respective file formats.
    They document the expected file formats in more detail.

    This class also provides methods to save the structures in LMDB or xyz format.

    :param file_path:
        Path to the file containing the structures.
        Supported formats are .lmdb, .xyz, and .csv.
    :type file_path: str
    :param property_keys:
        An optional sequence of property keys that should be read from the file and stored within the structures.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param floating_point_precision:
        Floating point precision to use for the structures. Can be one of "64-true", "64", "32-true", "32",
        "16-true", "16", "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
        Defaults to "64-true".
    :type floating_point_precision: Union[int, str, None]

    :raises KeyError:
        If any of the property keys are not found in the structure properties.
    """

    def __init__(self, file_path: str, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true") -> None:
        """Constructor for the StructureDataset class."""
        super().__init__()
        self._torch_precision = self._get_torch_precision(floating_point_precision)
        self._property_keys = list(property_keys) if property_keys is not None else []
        path = Path(file_path)

        if path.exists():
            existing_file_path = file_path
        else:
            # Try to use the data from the omg package.
            # noinspection PyTypeChecker
            existing_file_path = Path(files("omg").joinpath(file_path))
            if not existing_file_path.exists():
                raise FileNotFoundError(f"File path {file_path} neither exists on its own or within the omg package.")
        self._file_path = existing_file_path

        if path.suffix == ".lmdb":
            self._structures = self._from_lmdb(existing_file_path, property_keys)
        elif path.suffix == ".xyz":
            self._structures = self._from_xyz(existing_file_path, property_keys)
        elif path.suffix == ".csv":
            self._structures = self._from_csv(existing_file_path, property_keys)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats are .lmdb, .xyz, and .csv.")

        for structure in self._structures:
            structure.to(self._torch_precision)
            for property_key in self._property_keys:
                if property_key not in structure.property_dict:
                    raise KeyError(f"Property key '{property_key}' not found in structure properties.")

    @staticmethod
    def _get_torch_precision(floating_point_precision: Union[int, str, None] = "64-true") -> torch.dtype:
        """
        Get the torch precision based on the given floating point precision.

        :param floating_point_precision:
            Floating point precision to use. Can be one of "64-true", "64", "32-true", "32", "16-true", "16",
            "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
        :type floating_point_precision: Union[int, str, None]

        :return:
            The corresponding torch dtype.
        :rtype: torch.dtype

        :raises ValueError:
            If the given floating point precision is not recognized.
        """
        if floating_point_precision == "64-true" or floating_point_precision == "64" or floating_point_precision == 64:
            return torch.float64
        elif (floating_point_precision is None or floating_point_precision == "32-true" or
              floating_point_precision == "32" or floating_point_precision == 32 or
              floating_point_precision == "16-mixed" or floating_point_precision == "bf16-mixed"):
            return torch.float32
        elif (floating_point_precision == "16-true" or floating_point_precision == "16" or
              floating_point_precision == 16 or floating_point_precision == "transformer-engine-float16"):
            return torch.float16
        elif (floating_point_precision == "bf16-true" or floating_point_precision == "bf16" or
              floating_point_precision == "transformer-engine"):
            return torch.bfloat16
        else:
            raise ValueError(f"Unknown floating point precision: {floating_point_precision}")

    @staticmethod
    def _from_lmdb(existing_lmdb_path: Path, property_keys: Optional[Sequence[str]]) -> list[Structure]:
        """
        Load crystalline structures from an LMDB file.

        Each record (key) in the LMDB file is expected to contain a pickled dictionary with at least the following keys to
        represent a structure with N atoms:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the LMDB file from which the structure was loaded.
        - "file_key": The key in the LMDB file corresponding to the structure.
        - "identifier": If "ids" is present in the LMDB record, it will be stored here. This is usually a unique identifier
            of the structure, e.g., from Materials Project.

        It is possible to read additional property keys from the LMDB file into the structures by specifying them in the
        property_keys parameter. These keys will be read from the LMDB record and stored in the property_dict of the
        structure.

        :param existing_lmdb_path:
            Path to the LMDB file.
        :type existing_lmdb_path: Path
        :param property_keys:
            An optional sequence of property keys that are read from the LMDB file and stored within the structures.
            Defaults to None.
        :type property_keys: Optional[Sequence[str]]

        :return:
            A list of crystalline structures loaded from the LMDB file.
        :rtype: list[Structure]

        :raises KeyError:
            If any of the required keys are missing in the LMDB records.
        :raises TypeError:
            If any of the required keys have incorrect types in the LMDB records.
        """
        structures = []
        with (lmdb.Environment(str(existing_lmdb_path), subdir=False, readonly=True, lock=False) as env,
              env.begin() as txn):
            lmdb_configs = [(key.decode(), pickle.loads(data))
                            for key, data in tqdm(txn.cursor(), desc=f"Loading {existing_lmdb_path} data",
                                                  total=txn.stat()["entries"])]
            for key, lmdb_config in lmdb_configs:
                if "cell" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'cell'.")
                cell = lmdb_config["cell"]
                if not isinstance(cell, torch.Tensor):
                    raise TypeError(f"Key {key} in the lmdb file has 'cell' of type {type(cell)}, "
                                    f"expected torch.Tensor.")
                if not torch.is_floating_point(cell):
                    raise TypeError(f"Key {key} in the lmdb file has 'cell' of dtype {cell.dtype}, "
                                    f"expected floating point dtype.")

                if "atomic_numbers" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'atomic_numbers'.")
                atomic_numbers = lmdb_config["atomic_numbers"]
                if not isinstance(atomic_numbers, torch.Tensor):
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' of type {type(atomic_numbers)}, "
                                    f"expected torch.Tensor.")
                if not atomic_numbers.dtype is torch.int:
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' of dtype {atomic_numbers.dtype}, "
                                    f"expected torch.int.")
                atomic_numbers = atomic_numbers.tolist()

                if "pos" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'pos'.")
                pos = lmdb_config["pos"]
                if not isinstance(pos, torch.Tensor):
                    raise TypeError(f"Key {key} in the lmdb file has 'pos' of type {type(pos)}, "
                                    f"expected torch.Tensor.")
                if not torch.is_floating_point(pos):
                    raise TypeError(f"Key {key} in the lmdb file has 'pos' of dtype {pos.dtype}, "
                                    f"expected floating point dtype.")

                metadata = {"file_path": existing_lmdb_path, "file_key": key}
                if "ids" in lmdb_config:
                    metadata["identifier"] = lmdb_config["ids"]

                property_dict = {}
                if property_keys is not None:
                    for prop in property_keys:
                        if prop not in lmdb_config:
                            raise KeyError(f"Key {key} in the lmdb file does not contain '{prop}'.")
                        property_dict[prop] = lmdb_config[prop]

                structure = Structure(cell=cell, atomic_numbers=atomic_numbers, pos=pos, property_dict=property_dict,
                                      metadata=metadata)

                structures.append(structure)
        return structures

    @staticmethod
    def _from_csv(existing_csv_path: Path, property_keys: Optional[Sequence[str]]) -> list[Structure]:
        """
        Load crystalline structures from a CSV file.

        The CSV file is expected to contain a "cif" column with the CIF representation of the structures. This will be
        used to infer the cell, atomic numbers, and positions of the structures.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the CSV file from which the structure was loaded.
        - "file_key": The index in the CSV file corresponding to the structure.
        - "identifier": If a "material_id" column is present in the CSV file, it will be stored here. This is usually a
                        unique identifier of the structure, e.g., from Materials Project.

        It is possible to read additional property keys from the CSV file into the structures by specifying them in the
        property_keys parameter. These keys will be read from the CSV file and stored in the property_dict of the
        structure.

        :param existing_csv_path:
            Path to the CSV file.
        :type existing_csv_path: Path
        :param property_keys:
            An optional sequence of property keys that are read from the CSV file and stored within the structures.
            Defaults to None.
        :type property_keys: Optional[Sequence[str]]

        :return:
            A list of crystalline structures loaded from the CSV file.
        :rtype: list[Structure]

        :raises KeyError:
            If any of the required keys are missing in the CSV records.
        """
        csv_structures = pd.read_csv(existing_csv_path)
        if "cif" not in csv_structures.keys():
            raise KeyError(f"CSV file does not contain 'cif' column.")
        if property_keys is not None:
            for prop in property_keys:
                if prop not in csv_structures.keys():
                    raise KeyError(f"CSV file does not contain '{prop}' column.")

        structures = []
        for structure_index in tqdm(range(len(csv_structures)), desc=f"Loading {existing_csv_path} data"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pymatgen_structure = PymatgenStructure.from_str(csv_structures["cif"][structure_index], fmt="cif")
            # TODO: CAREFULLY MAKE SURE THAT ALL THE STRUCTURES ARE MATCHING AND THAT ALL LATTICES AGREE.

            pos = torch.tensor(pymatgen_structure.cart_coords)
            cell = torch.tensor(pymatgen_structure.lattice.matrix)
            atomic_numbers = list(pymatgen_structure.atomic_numbers)

            metadata = {"file_path": existing_csv_path, "file_key": structure_index}
            if "material_id" in csv_structures.keys():
                metadata["identifier"] = csv_structures["material_id"][structure_index]

            property_dict = {}
            if property_keys is not None:
                for prop in property_keys:
                    property_dict[prop] = torch.tensor(csv_structures[prop][structure_index])
            structure = Structure(cell=cell, atomic_numbers=atomic_numbers, pos=pos, property_dict=property_dict,
                                  metadata=metadata)
            structures.append(structure)
        return structures

    def to_lmdb(self, lmdb_path: Path) -> None:
        """
        Save the structures to an LMDB file.

        :param lmdb_path:
            Path to the LMDB file.
        :type lmdb_path: Path

        :raises FileExistsError:
            If the LMDB file already exists.
        """
        if lmdb_path.exists():
            raise FileExistsError(f"LMDB path {lmdb_path} already exists.")

        with (lmdb.Environment(str(lmdb_path), subdir=False, map_size=int(1e12), lock=False) as env,
              env.begin(write=True) as txn):
            for idx, structure in tqdm(enumerate(self._structures), desc=f"Saving configurations to {lmdb_path}",
                                    total=len(self._structures)):
                data = {
                    "pos": structure.pos,
                    "cell": structure.cell,
                    "atomic_numbers": structure.atomic_numbers,
                } | structure.property_dict | structure.metadata
                txn.put(str(idx).encode(), pickle.dumps(data))

    def to_xyz(self, xyz_path: Path) -> None:
        """
        Save the structures to an XYZ file.

        :param xyz_path:
            Path to the XYZ file.
        :type xyz_path: Path

        :raises FileExistsError:
            If the XYZ file already exists.
        """
        if xyz_path.exists():
            raise FileExistsError(f"XYZ path {xyz_path} already exists.")
        all_atoms = []
        for struc in tqdm(self._structures, desc=f"Converting configurations to ASE Atoms"):
            # TODO: TEST!
            all_atoms.append(Atoms(numbers=struc.atomic_numbers, positions=struc.pos.numpy(), cell=struc.cell, pbc=True,
                                   info=struc.property_dict))
        write(str(xyz_path), all_atoms, format="extxyz")

    def __len__(self) -> int:
        """
        Get the number of structures within the dataset.

        This method is required to make the dataset compatible with PyTorch dataloaders.

        :return:
        The number of structures within the dataset.
        :rtype: int
        """
        return len(self._structures)

    def __getitem__(self, idx: int) -> Structure:
        """
        Return the structure at the given index.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure
        """
        return self._structures[idx]


class OverfittingDataset(StructureDataset):
    """
    Dataset that always returns the same crystalline structure from an LMDB file.

    This dataset is useful for overfitting tests. It expects the same LMDB file format as LmdbDataset.

    :param lmdb_path:
        Path to the LMDB file.
        This can either be an absolute path, a relative path to the current working directory, or
        a relative path within the omg package.
    :type lmdb_path: str
    :param structure_index:
        Index of the structure in the LMDB file to always return.
        Defaults to 0.
    :type structure_index: int
    :param property_keys:
        An optional sequence of property keys that are read from the LMDB file and stored within the
        structures.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param floating_point_precision:
        Floating point precision to use in the structures. Can be one of "64-true", "64", "32-true", "32", "16-true",
        "16", "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
        Defaults to "64-true".
    :type floating_point_precision: Union[int, str, None]

    :raises KeyError:
        If the structure_index is out of bounds.
    """

    def __init__(self, lmdb_path: str, structure_index: int = 0, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true") -> None:
        """Constructor for the OverfittingDataset class."""
        super().__init__(lmdb_path=lmdb_path, property_keys=property_keys,
                         floating_point_precision=floating_point_precision)
        if not 0 <= structure_index < len(self):
            raise KeyError(f"Invalid structure index {structure_index}, "
                           f"possible values are 0 to {len(self) - 1}.")
        self._structure_index = structure_index

    def __getitem__(self, idx: int) -> Structure:
        """
        Return the structure at the given index.

        This method ignores the given index and always return the same configuration.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure
        """
        return super().__getitem__(self._structure_index)


if __name__ == '__main__':
    dataset = LmdbDataset(lmdb_path="data/mp_20/test.lmdb")
