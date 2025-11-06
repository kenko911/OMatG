import hashlib
from importlib.resources import files
from pathlib import Path
import pickle
from typing import Any, Optional, Sequence, Union
import warnings
from ase import Atoms
from ase.io import write
from ase.symbols import Symbols
import lmdb
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
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

    @classmethod
    def from_dictionary(cls, data: dict[str, torch.Tensor], property_keys: Sequence[str],
                        metadata: dict[str, Any]) -> "Structure":
        """
        Create a Structure object from the given data dictionary and metadata.

        The data dictionary is expected to contain at least the following keys:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        Additionally, the data dictionary should contain additional property keys specified in the property_keys
        sequence.

        :param data:
            A dictionary representing the structure.
        :type data: dict[str, torch.Tensor]
        :param property_keys:
            A sequence of property keys to be included in the Structure's property_dict.
        :type property_keys: Sequence[str]
        :param metadata:
            A dictionary of metadata to be included in the Structure's metadata.
        :type metadata: dict[str, Any]

        :return:
            A Structure object created from the data dictionary and metadata.
        :rtype: Structure
        """
        return cls(
            cell=data["cell"],
            atomic_numbers=data["atomic_numbers"].tolist(),
            pos=data["pos"],
            property_dict={prop: data[prop] for prop in property_keys},
            metadata=metadata
        )

    def to_dictionary(self) -> dict[str, Any]:
        """
        Return the structure as a data dictionary.

        The returned dictionary contains the following keys:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
        - "pos": A Nx3 torch.Tensor of atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        Additionally, the dictionary includes all properties from the property_dict and metadata.

        :return:
            A data dictionary representing the structure.
        :rtype: dict[str, Any]
        """
        return { # TODO: REPLACE ATOMIC_NUMBERS EVERYWHERE BY TORCH TENSOR
            "pos": self.pos,
            "cell": self.cell,
            "atomic_numbers": torch.tensor(self.atomic_numbers, dtype=torch.int32)
        } | self.property_dict | self.metadata

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
    def symbols(self) -> list[str]:
        """
        Return the chemical symbols of the atoms in the structure.

        :return:
            A list of N strings giving the chemical symbols of the atoms, where N is the number of atoms.
        :rtype: list[str]
        """
        return list(Symbols(self._atomic_numbers))

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

    def get_ase_atoms(self) -> Atoms:
        """
        Convert the structure to an ASE Atoms object.

        :return:
            The ASE Atoms object.
        :rtype: Atoms
        """
        # noinspection PyTypeChecker
        return Atoms(numbers=self.atomic_numbers, positions=self.pos.numpy(), cell=self.cell.numpy(), pbc=True,
                     info=self.property_dict | self.metadata)

    def get_pymatgen_structure(self) -> PymatgenStructure:
        """
        Convert the structure to a pymatgen Structure object.

        :return:
            The pymatgen Structure object.
        :rtype: PymatgenStructure
        """
        return PymatgenStructure(
            lattice=PymatgenLattice(self.cell.numpy()),
            species=self.symbols,
            coords=self.pos.numpy(),
            coords_are_cartesian=True,
            properties=self.property_dict | self.metadata
        )


class StructureDataset(Dataset):
    """
    Dataset for reading crystalline structures from several file formats.

    This dataset reads the structures lazily from a file when they are requested. This is achieved with the help of
    the LMDB database format, which allows for efficient key-value storage and retrieval.

    The supported file formats for the storage of crystalline structures are LMDB and CSV. The specific format is
    inferred from the file extension.

    The _from_lmdb and _from_csv methods are used to read the structures from the respective file formats on
    initialization of this class. The _from_lmdb basically serves as a test case for the LMDB reading functionality,
    while the _from_csv method reads structures from a CSV file containing CIF strings and converts them to the LMDB
    format for efficient access. The LMDB cache file is created in the same directory as the CSV file and is named based
    on the hash of the CSV file and the preprocessing options used. If the cache file already exists, it is used
    directly instead of creating a new one.

    The _from_lmdb and _from_csv methods document the expected file contents for their respective file types in more
    detail. Additional parser keyword arguments of the _from_lmdb and _from_csv methods can be passed via parser_kwargs.

    This class also provides methods to save all the structures in LMDB or xyz format.

    :param file_path:
        Path to the file containing the structures.
        This can either be an absolute path, a relative path to the current working directory, or a relative path within
        the omg package.
        Supported formats are .lmdb and .csv.
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
    :param parser_kwargs:
        Additional keyword arguments to pass to the specific file format parser methods.
    :type parser_kwargs: Any

    :raises KeyError:
        If any of the property keys are not found in the structure properties.
        If reserved property keys are used (i.e., "cell", "pos", "atomic_numbers", "file_path", "file_key", or
        "identifier").
    """

    def __init__(self, file_path: str, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true", **parser_kwargs: Any) -> None:
        """Constructor for the StructureDataset class."""
        super().__init__()
        self._torch_precision = self._get_torch_precision(floating_point_precision)
        self._property_keys = list(property_keys) if property_keys is not None else []
        for prop in self._property_keys:
            if prop in ("cell", "pos", "atomic_numbers", "file_path", "file_key", "identifier"):
                raise KeyError(f"Property key '{prop}' is reserved and cannot be used as a property key.")
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
            # noinspection PyArgumentList
            self._file, self._number_structures = self._from_lmdb(self._file_path, self._property_keys, **parser_kwargs)
        elif path.suffix == ".csv":
            self._file, self._number_structures = self._from_csv(self._file_path, self._property_keys, **parser_kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats are .lmdb and .csv.")

        # Read structures lazily from this file.
        self._env = lmdb.Environment(str(self._file), subdir=False, readonly=True, lock=False, readahead=False,
                                     meminit=False)

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
    def _compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """
        Compute the hash of a file using the specified algorithm.

        Possible algorithms are those supported by hashlib.file_digest, i.e., typically "md5", "sha1", "sha224",
        "sha256", "sha384", "sha512", "sha3_224", "sha3_256", "sha3_384", "sha3_512", "shake_128", "shake_256",
        "blake2b", and "blake2s".

        :param file_path:
            Path to the file.
        :type file_path: Path
        :param algorithm:
            Hashing algorithm to use. Defaults to "sha256".
        :type algorithm: str

        :return:
            The hexadecimal digest of the file hash.
        :rtype: str

        :raises ValueError:
            If the specified algorithm is not supported.
        """
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hashing algorithm: {algorithm}. "
                             f"Supported algorithms are: {hashlib.algorithms_available}.")
        with open(file_path, "rb") as f:
            digest = hashlib.file_digest(f, algorithm)
        return digest.hexdigest()

    @staticmethod
    def _from_lmdb(existing_lmdb_path: Path, property_keys: Sequence[str]) -> tuple[Path, int]:
        """
        Check proper format of the LMDB file for crystalline structures and return its path and number of structures.

        The records (keys) in the LMDB file are expected to be encoded integers starting from 0 to N_S-1, where N_S is
        the number of structures in the file.

        Each record (key) in the LMDB file is expected to contain a pickled dictionary with at least the following keys
        to represent a structure with N atoms:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the LMDB file from which the structure was loaded.
        - "file_key": The key in the LMDB file corresponding to the structure.
        - "identifier": If "ids" or "identifier" is present in the LMDB record, it will be stored here. This is usually
                        a unique identifier of the structure, e.g., from Materials Project.

        It is possible to read additional property keys from the LMDB file into the structures by specifying them in the
        property_keys parameter. These keys will be read from the LMDB record and stored in the property_dict of the
        structure.

        :param existing_lmdb_path:
            Path to the LMDB file.
        :type existing_lmdb_path: Path
        :param property_keys:
            A sequence of property keys that are read from the LMDB file and stored within the structures.
        :type property_keys: Sequence[str]

        :return:
            The path of the LMDB file, the number of structures in the LMDB file.
        :rtype: tuple[Path, int]

        :raises KeyError:
            If any of the required keys are missing in the LMDB records.
        :raises TypeError:
            If any of the required keys have incorrect types in the LMDB records.
        """
        with (lmdb.Environment(str(existing_lmdb_path), subdir=False, readonly=True, lock=False) as env,
              env.begin() as txn):
            number_structures = txn.stat()["entries"]

            for int_key in tqdm(range(number_structures), desc=f"Loading {existing_lmdb_path} data"):
                data = txn.get(str(int_key).encode())
                lmdb_structure = pickle.loads(data)

                if "cell" not in lmdb_structure:
                    raise KeyError(f"Key {int_key} in the lmdb file does not contain 'cell'.")
                if not isinstance(lmdb_structure["cell"], torch.Tensor):
                    raise TypeError(f"Key {int_key} in the lmdb file has 'cell' of type "
                                    f"{type(lmdb_structure["cell"])}, expected torch.Tensor.")
                if not torch.is_floating_point(lmdb_structure["cell"]):
                    raise TypeError(f"Key {int_key} in the lmdb file has 'cell' of "
                                    f"dtype {lmdb_structure["cell"].dtype}, expected floating point dtype.")

                if "atomic_numbers" not in lmdb_structure:
                    raise KeyError(f"Key {int_key} in the lmdb file does not contain 'atomic_numbers'.")
                if not isinstance(lmdb_structure["atomic_numbers"], torch.Tensor):
                    raise TypeError(f"Key {int_key} in the lmdb file has 'atomic_numbers' of type "
                                    f"{type(lmdb_structure["atomic_numbers"])}, expected torch.Tensor.")
                if not lmdb_structure["atomic_numbers"].dtype is torch.int:
                    raise TypeError(f"Key {int_key} in the lmdb file has 'atomic_numbers' of dtype "
                                    f"{lmdb_structure["atomic_numbers"].dtype}, expected torch.int.")

                if "pos" not in lmdb_structure:
                    raise KeyError(f"Key {int_key} in the lmdb file does not contain 'pos'.")
                if not isinstance(lmdb_structure["pos"], torch.Tensor):
                    raise TypeError(f"Key {int_key} in the lmdb file has 'pos' of type "
                                    f"{type(lmdb_structure["pos"])}, expected torch.Tensor.")
                if not torch.is_floating_point(lmdb_structure["pos"]):
                    raise TypeError(f"Key {int_key} in the lmdb file has 'pos' of dtype "
                                    f"{lmdb_structure["pos"].dtype}, expected floating point dtype.")

                metadata = {"file_path": str(existing_lmdb_path), "file_key": int_key}
                if "ids" in lmdb_structure and "identifier" in lmdb_structure:
                    raise KeyError(f"Key {int_key} in the lmdb file contains both 'ids' and 'identifier'. Only one of "
                                   f"these should be present.")
                # Included for backwards compatibility.
                if "ids" in lmdb_structure:
                    metadata["identifier"] = lmdb_structure["ids"]
                if "identifier" in lmdb_structure:
                    metadata["identifier"] = lmdb_structure["identifier"]

                for prop in property_keys:
                    if prop not in lmdb_structure:
                        raise KeyError(f"Key {int_key} in the lmdb file does not contain '{prop}'.")

                # Test whether the structure can be created.
                Structure.from_dictionary(lmdb_structure, property_keys, metadata)

        return existing_lmdb_path, number_structures

    @staticmethod
    def _from_csv(existing_csv_path: Path, property_keys: Sequence[str],
                  cdvae_preprocessing: bool = True, mattergen_preprocessing: bool = False) -> tuple[Path, int]:
        """
        Check proper format of the CSV file for crystalline structures, convert to LMDB, and return the LMDB path and
        number of structures.

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

        The structures from the CSV files can optionally be preprocessed to match the conventions used in CDVAE (that
        are also used in DiffCSP and FlowMM) or MatterGen.

        The structures are converted to the LMDB format for efficient access. The LMDB cache file is created in the same
        directory as the CSV file and is named based on the hash of the CSV file and the preprocessing options used.
        If the cache file already exists, it is used directly instead of creating a new one.

        :param existing_csv_path:
            Path to the CSV file.
        :type existing_csv_path: Path
        :param property_keys:
            A sequence of property keys that are read from the CSV file and stored within the structures.
        :type property_keys: Sequence[str]
        :param cdvae_preprocessing:
            Whether to preprocess the structures to match the conventions used in CDVAE.
            This applies the Niggli reduction of PyMatgen.
            Defaults to True.
        :type cdvae_preprocessing: bool
        :param mattergen_preprocessing:
            Whether to preprocess the structures to match the conventions used in MatterGen.
            This applies the primitive cell extraction followed by the Niggli reduction of PyMatgen.
            Defaults to False.
        :type mattergen_preprocessing: bool

        :return:
            The path of the LMDB cache file, the number of structures in the LMDB file.
        :rtype: tuple[Path, int]

        :raises ValueError:
            If both cdvae_preprocessing and mattergen_preprocessing are enabled at the same time.
        :raises KeyError:
            If any of the required keys are missing in the CSV records.
        """
        if cdvae_preprocessing and mattergen_preprocessing:
            raise ValueError("CDVAE and MatterGen preprocessing cannot both be enabled at the same time.")

        # Find or create cache file.
        # The cache file is identified by the hash of the CSV file and the preprocessing options.
        file_hash = StructureDataset._compute_file_hash(existing_csv_path, algorithm="sha256")
        cache_parts = [file_hash]
        if cdvae_preprocessing:
            cache_parts.append("cdvae")
        if mattergen_preprocessing:
            cache_parts.append("mattergen")
        if property_keys is not None:
            for prop in sorted(property_keys):
                cache_parts.append(prop)
        cache_identifier = "_".join(cache_parts)
        cache_file = existing_csv_path.with_suffix(f".csv.{cache_identifier}.lmdb")

        if cache_file.exists():
            return StructureDataset._from_lmdb(cache_file, property_keys=property_keys)

        # Check for correct columns.
        csv_columns = pd.read_csv(existing_csv_path, nrows=0).columns
        if "cif" not in csv_columns:
            raise KeyError(f"CSV file does not contain 'cif' column.")
        if property_keys is not None:
            for prop in property_keys:
                if prop not in csv_columns:
                    raise KeyError(f"CSV file does not contain '{prop}' column.")

        # Make sure to only read one row at a time to avoid reading the entire file into memory.
        csv_structures = pd.read_csv(existing_csv_path, chunksize=1)
        number_structures = 0
        changed_structures = 0
        with (lmdb.Environment(str(cache_file), subdir=False, map_size=int(1e12), lock=False) as env,
              env.begin(write=True) as txn):
            for structure_index, structure_chunk in tqdm(enumerate(csv_structures),
                                                         desc=f"Loading {existing_csv_path} data",
                                                         unit=" structures"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pymatgen_structure = PymatgenStructure.from_str(structure_chunk["cif"][structure_index], fmt="cif")

                # See https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils.py.
                # Configurations at https://github.com/txie-93/cdvae/tree/main/conf/data are such that niggli=True and
                # primitive=False.
                if cdvae_preprocessing:
                    cdvae_structure = pymatgen_structure.get_reduced_structure()
                    # This rotates the lattice but since we use fractional coordinates, it doesn't really matter.
                    cdvae_structure = PymatgenStructure(
                        lattice=PymatgenLattice.from_parameters(*cdvae_structure.lattice.parameters),
                        species=cdvae_structure.species,
                        coords=cdvae_structure.frac_coords,
                        coords_are_cartesian=False,
                    )
                    # Verify that the structures match.
                    if StructureMatcher(ltol=1e-4, angle_tol=1e-4, stol=1e-4, scale=False).get_rms_dist(
                            cdvae_structure, pymatgen_structure) is None:
                        changed_structures += 1
                    pymatgen_structure = cdvae_structure

                # See https://github.com/microsoft/mattergen/blob/main/mattergen/common/data/dataset.py.
                if mattergen_preprocessing:
                    # Note that this is also effectively called when one passes primitive=True to from_str.
                    mattergen_structure = pymatgen_structure.get_primitive_structure()
                    mattergen_structure = mattergen_structure.get_reduced_structure()
                    # Verify that the structures match.
                    if StructureMatcher(ltol=1e-4, angle_tol=1e-4, stol=1e-4, scale=False).get_rms_dist(
                            mattergen_structure, pymatgen_structure) is None:
                        changed_structures += 1
                    pymatgen_structure = mattergen_structure

                pos = torch.tensor(pymatgen_structure.cart_coords)
                cell = torch.tensor(pymatgen_structure.lattice.matrix)
                atomic_numbers = list(pymatgen_structure.atomic_numbers)

                metadata = {"file_path": str(cache_file), "file_key": structure_index}
                if "material_id" in csv_columns:
                    metadata["identifier"] = structure_chunk["material_id"][structure_index]

                property_dict = {}
                for prop in property_keys:
                    property_dict[prop] = torch.tensor(structure_chunk[prop][structure_index])

                structure = Structure(cell=cell, atomic_numbers=atomic_numbers, pos=pos, property_dict=property_dict,
                                      metadata=metadata)

                lmdb_structure = structure.to_dictionary()
                txn.put(str(structure_index).encode(), pickle.dumps(lmdb_structure))

                number_structures += 1

        if changed_structures > 0:
            warnings.warn(f"{changed_structures} structures were changed during preprocessing.")

        return cache_file, number_structures

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
            for idx in tqdm(range(self._number_structures), desc=f"Saving structures to {lmdb_path}"):
                structure = self[idx]
                lmdb_structure = structure.to_dictionary()
                txn.put(str(idx).encode(), pickle.dumps(lmdb_structure))

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

        for idx in tqdm(range(self._number_structures), desc=f"Saving structures to {xyz_path}"):
            structure = self[idx]
            atoms = structure.get_ase_atoms()
            write(str(xyz_path), atoms, format="extxyz", append=True)

    def __len__(self) -> int:
        """
        Get the number of structures within the dataset.

        This method is required to make the dataset compatible with PyTorch dataloaders.

        :return:
        The number of structures within the dataset.
        :rtype: int
        """
        return self._number_structures

    def __getitem__(self, idx: int) -> Structure:
        """
        Return the structure at the given index.

        The structure is read lazily from the LMDB file of this class at the given key.

        Each record (key) in the LMDB file is expected to contain a pickled dictionary with at least the following keys
        to represent a structure with N atoms:
        - "cell": A 3x3 torch.Tensor of the lattice vectors.
                  The [i, j]-th element is the jth Cartesian coordinate of the ith unit vector.
        - "pos": A Nx3 torch.Tensor of atomic positions.
        - "atomic_numbers": A torch tensor of N integers giving the atomic numbers of the atoms.

        The metadata of each structure will contain the following keys:
        - "file_path": The path to the LMDB file from which the structure was loaded.
        - "file_key": The key in the LMDB file corresponding to the structure.
        - "identifier": If "ids" or "identifier" is present in the LMDB record, it will be stored here. This is usually
                        a unique identifier of the structure, e.g., from Materials Project.

        Additionally, the property keys will be read from the LMDB record and stored in the property_dict of the
        structure.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure

        :raises IndexError:
            If the given index is out of bounds.
        """
        if idx < 0 or idx >= self._number_structures:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self._number_structures}.")

        with self._env.begin(write=False) as txn:
            lmdb_structure = pickle.loads(txn.get(str(idx).encode()))

        metadata = {"file_path": str(self._file_path), "file_key": idx}
        assert not ("ids" in lmdb_structure and "identifier" in lmdb_structure)
        if "ids" in lmdb_structure:
            metadata["identifier"] = lmdb_structure["ids"]
        if "identifier" in lmdb_structure:
            metadata["identifier"] = lmdb_structure["identifier"]

        structure = Structure.from_dictionary(lmdb_structure, self._property_keys, metadata)
        structure.to(self._torch_precision)

        return structure

    def __getstate__(self) -> dict[str, Any]:
        """
        Return the state of the object for pickling.

        This method removes the LMDB environment from the state to avoid pickling issues. This is required for
        dataloaders with multiple workers.

        :return:
            The state of the object.
        :rtype: dict[str, Any]
        """
        state = self.__dict__.copy()
        # Remove the LMDB environment from the state to avoid pickling issues.
        state["_env"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set the state of the object from pickling.

        This method restores the LMDB environment from the state after unpickling. This is required for dataloaders
        with multiple workers.

        :param state:
            The state of the object.
        :type state: dict[str, Any]
        """
        self.__dict__.update(state)
        self._env = lmdb.Environment(str(self._file), subdir=False, readonly=True, lock=False, readahead=False,
                                     meminit=False)

    def __del__(self) -> None:
        """
        Close the LMDB environment when the object is deleted.
        """
        if hasattr(self, "_env"):
            self._env.close()


class OverfittingDataset(StructureDataset):
    """
    Dataset that always returns the same crystalline structure from a file.

    This dataset is useful for overfitting tests. It expects the same LMDB or CSV file formats as StructureDataset.

    :param file_path:
        Path to the file containing the structures.
        This can either be an absolute path, a relative path to the current working directory, or a relative path within
        the omg package.
        Supported formats are .lmdb and .csv.
    :type file_path: str
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
    :param parser_kwargs:
        Additional keyword arguments to pass to the specific file format parser methods.
    :type parser_kwargs: Any

    :raises KeyError:
        If the structure_index is out of bounds.
    """

    def __init__(self, file_path: str, structure_index: int = 0, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true", **parser_kwargs: Any) -> None:
        """Constructor for the OverfittingDataset class."""
        super().__init__(file_path=file_path, property_keys=property_keys,
                         floating_point_precision=floating_point_precision, **parser_kwargs)
        if not 0 <= structure_index < len(self):
            raise KeyError(f"Invalid structure index {structure_index}, "
                           f"possible values are 0 to {len(self) - 1}.")
        self._structure_index = structure_index

    def __getitem__(self, idx: int) -> Structure:
        """
        Return the structure at the given index.

        This method ignores the given index and always returns the same structure.

        :param idx:
            Index of the structure to return.
        :type idx: int

        :return:
            The structure at the given index.
        :rtype: Structure
        """
        return super().__getitem__(self._structure_index)


if __name__ == '__main__':
    store = False
    dataset_csv = StructureDataset(file_path="data/mp_20/test.csv", cdvae_preprocessing=False,
                                   mattergen_preprocessing=False)
    dataset_csv_cdvae = StructureDataset(file_path="data/mp_20/test.csv", cdvae_preprocessing=True,
                                         mattergen_preprocessing=False)
    dataset_csv_mattergen = StructureDataset(file_path="data/mp_20/test.csv", cdvae_preprocessing=False,
                                             mattergen_preprocessing=True)
    dataset_lmdb = StructureDataset(file_path="data/mp_20/test.lmdb")
    overfitting = OverfittingDataset(file_path="data/mp_20/test.lmdb", structure_index=0)

    if store:
        dataset_csv.to_xyz(Path("csv_test_new.xyz"))
        dataset_csv_mattergen.to_xyz(Path("mattergen_test_new.xyz"))
        dataset_csv_cdvae.to_xyz(Path("cdvae_test_new.xyz"))
        dataset_lmdb.to_xyz(Path("lmdb_test_new.xyz"))

    # noinspection PyTypeChecker
    for csv in tqdm(dataset_csv):
        lmdbs = [lmdb for lmdb in dataset_lmdb if lmdb.metadata["identifier"] == csv.metadata["identifier"]]
        assert len(lmdbs) == 1
        lmdb = lmdbs[0]
        assert all(ca == la for ca, la in zip(csv.atomic_numbers, lmdb.atomic_numbers))
        csv_structure = csv.get_pymatgen_structure()
        lmdb_structure = lmdb.get_pymatgen_structure()
        sm = StructureMatcher(ltol=1e-3, angle_tol=1e-3, stol=1e-3, scale=False)
        res = sm.get_rms_dist(csv_structure, lmdb_structure)
        assert res is not None

    # noinspection PyTypeChecker
    for csv in tqdm(dataset_csv_cdvae):
        lmdbs = [lmdb for lmdb in dataset_lmdb if lmdb.metadata["identifier"] == csv.metadata["identifier"]]
        assert len(lmdbs) == 1
        lmdb = lmdbs[0]
        assert all(ca == la for ca, la in zip(csv.atomic_numbers, lmdb.atomic_numbers))
        assert torch.allclose(csv.cell, lmdb.cell)
        csv_structure = csv.get_pymatgen_structure()
        lmdb_structure = lmdb.get_pymatgen_structure()
        sm = StructureMatcher(ltol=1e-3, angle_tol=1e-3, stol=1e-3, scale=False)
        res = sm.get_rms_dist(csv_structure, lmdb_structure)
        assert res is not None
