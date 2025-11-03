from importlib.resources import files
from pathlib import Path
import pickle
from typing import Any, Optional, Sequence, Union
from ase import Atoms
from ase.io import write
import lmdb
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
    Dataset for storing a list of crystalline structures.

    This class provides methods to save the structures in LMDB or xyz format.

    :param structures:
        A sequence of crystalline structures to store in the dataset.
    :type structures: Sequence[Structure]
    :param property_keys:
        An optional sequence of property keys that are stored within the structures.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param floating_point_precision:
        Floating point precision to use in the structures. Can be one of "64-true", "64", "32-true", "32",
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
        Dataset of crystalline structures that are loaded from an LMDB file.

        Each record (key) in the LMDB file is expected to contain a pickled dictionary with at least the following keys to
        represent a structure with N atoms:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of atomic positions.
        - "atomic_numbers": A list of N integers giving the atomic numbers of the atoms.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the LMDB file from which the structure was loaded.
        - "file_key": The key in the LMDB file corresponding to the structure.
        - "identifier": If "ids" is present in the LMDB record, it will be stored here. This is usually a unique identifier
            of the structure, e.g., from Materials Project.

        It is possible to read additional property keys from the LMDB file into the structures by specifying them in the
        property_keys parameter. These keys will be read from the LMDB record and stored in the property_dict of the
        structure.

        :param lmdb_path:
            Path to the LMDB file.
            This can either be an absolute path, a relative path to the current working directory, or a relative path
            within the omg package.
        :type lmdb_path: str
        :param property_keys:
            An optional sequence of property keys that are read from the LMDB file and stored within the structures.
            Defaults to None.
        :type property_keys: Optional[Sequence[str]]
        :param floating_point_precision:
            Floating point precision to use in the structures. Can be one of "64-true", "64", "32-true", "32",
            "16-true", "16", "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
            Defaults to "64-true".
        :type floating_point_precision: Union[int, str, None]

        :raises FileNotFoundError:
            If the LMDB file does not exist.
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
                if not isinstance(lmdb_config["atomic_numbers"], list):
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' of type {type(atomic_numbers)}, "
                                    f"expected list[int].")
                if not all(isinstance(num, int) for num in atomic_numbers):
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' containing elements of types "
                                    f"{set(type(num) for num in atomic_numbers)}, expected list[int].")

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


class LmdbDataset(StructureDataset):
    """
    Dataset of crystalline structures that are loaded from an LMDB file.

    Each record (key) in the LMDB file is expected to contain a pickled dictionary with at least the following keys to
    represent a structure with N atoms:
    - "cell": A 3x3 torch.Tensor of the lattice vectors.
              The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
    - "pos": A Nx3 torch.Tensor of atomic positions.
    - "atomic_numbers": A list of N integers giving the atomic numbers of the atoms.

    The metadata of each structure will contain the following keys:
    - "file_path": The path to the LMDB file from which the structure was loaded.
    - "file_key": The key in the LMDB file corresponding to the structure.
    - "identifier": If "ids" is present in the LMDB record, it will be stored here. This is usually a unique identifier
        of the structure, e.g., from Materials Project.

    It is possible to read additional property keys from the LMDB file into the structures by specifying them in the
    property_keys parameter. These keys will be read from the LMDB record and stored in the property_dict of the
    structure.

    :param lmdb_path:
        Path to the LMDB file.
        This can either be an absolute path, a relative path to the current working directory, or a relative path
        within the omg package.
    :type lmdb_path: str
    :param property_keys:
        An optional sequence of property keys that are read from the LMDB file and stored within the structures.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param floating_point_precision:
        Floating point precision to use in the structures. Can be one of "64-true", "64", "32-true", "32",
        "16-true", "16", "bf16-true", "bf16", "transformer-engine-float16", or "transformer-engine".
        Defaults to "64-true".
    :type floating_point_precision: Union[int, str, None]

    :raises FileNotFoundError:
        If the LMDB file does not exist.
    """

    def __init__(self, lmdb_path: str, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true") -> None:
        """Constructor for the LmdbDataset class."""
        path = Path(lmdb_path)

        if path.exists():
            existing_lmdb_path = lmdb_path
        else:
            # Try to use the data from the omg package.
            # noinspection PyTypeChecker
            existing_lmdb_path = Path(files("omg").joinpath(lmdb_path))
            if not existing_lmdb_path.exists():
                raise FileNotFoundError(f"LMDB path {lmdb_path} neither exists on its own or within the omg package.")

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
                if not isinstance(lmdb_config["atomic_numbers"], list):
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' of type {type(atomic_numbers)}, "
                                    f"expected list[int].")
                if not all(isinstance(num, int) for num in atomic_numbers):
                    raise TypeError(f"Key {key} in the lmdb file has 'atomic_numbers' containing elements of types "
                                    f"{set(type(num) for num in atomic_numbers)}, expected list[int].")

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

        super().__init__(structures=structures, property_keys=property_keys,
                         floating_point_precision=floating_point_precision)


class CsvDataset(ConfigurationDataset):
    def __init__(self, csv_path: Path, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true"):
        """
        Read configurations from a CIF file and initialize a dataset.

        Args:
            cif_path: Path to the CIF file.
            property_keys: List of property keys to read from the CIF file.
            floating_point_precision: Floating point precision to use for the properties.

        Returns:
            A dataset of configurations.
        """
        torch_precision = self._get_torch_precision(floating_point_precision)

        if csv_path.exists():
            existing_csv_path = csv_path
        else:
            # Try to use the data from the omg package.
            existing_csv_path = Path(files("omg").joinpath(csv_path))
            if not existing_csv_path.exists():
                raise FileNotFoundError(f"CSV path {csv_path} neither exists on its own or within the omg package.")

        csv_configs = pd.read_csv(existing_csv_path)
        if "cif" not in csv_configs:
            raise KeyError(f"CSV file does not contain 'cif' column.")
        if "material_id" not in csv_configs:
            raise KeyError(f"CSV file does not contain 'material_id' column.")
        if property_keys is not None:
            for prop in property_keys:
                if prop not in csv_configs:
                    raise KeyError(f"CSV file does not contain '{prop}' column.")

        configs = []
        for config_index in tqdm(range(len(csv_configs)), desc=f"Loading {existing_csv_path} data"):
            #crystal_likely = crystal.get_reduced_structure()

            #crystal_likely = Structure(
            #  lattice=Lattice.from_parameters(*crystal_likely.lattice.parameters),
            #    species=crystal_likely.species,
            #    coords=crystal_likely.frac_coords,
            #    coords_are_cartesian=False,
            #)


            structure = Structure.from_str(csv_configs["cif"][config_index], fmt="cif", primitive=True)
            structure = structure.get_reduced_structure()
            structure = Structure(
                lattice=Lattice.from_parameters(*structure.lattice.parameters),
                species=structure.species,
                coords=structure.frac_coords,
                coords_are_cartesian=False,
            )
            pos = torch.tensor(structure.lattice.get_cartesian_coords(structure.frac_coords), dtype=torch_precision)
            cell = torch.tensor(structure.lattice.as_dict()["matrix"], dtype=torch_precision)
            species = [chemical_symbols[int(i)] for i in structure.atomic_numbers]
            pbc = structure.lattice.pbc
            ids = csv_configs["material_id"][config_index]

            property_dict = {}
            if property_keys is not None:
                for prop in property_keys:
                    property_dict[prop] = torch.tensor(csv_configs[prop][config_index])
                    if property_dict[prop].is_floating_point():
                        property_dict[prop] = property_dict[prop].to(torch_precision)
            config = Configuration(cell=cell, species=species, pos=pos, pbc=pbc, ids=ids,
                                   property_dict=property_dict,
                                   metadata={"file_path": existing_csv_path, "file_key": config_index})
            configs.append(config)
        super().__init__(configurations=configs, property_keys=property_keys)




class OverfittingDataset(LmdbDataset):
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
