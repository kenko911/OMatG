from importlib.resources import files
from pathlib import Path
import pickle
from typing import Any, Optional, Sequence, Union
from ase import Atoms
from ase.data import chemical_symbols
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Structure(object):
    """
    Storage for a single crystalline structure of atoms.

    The structure is represented by its cell, atom species, and real atom coordinates.  # TODO: This should probably be fractional.
    Additionally, one can store arbitrary properties and metadata associated with the structure in this class.

    :param cell:
        A 3x3 matrix of the lattice vectors. The [i, j]-th element is the jth Cartesian coordinate of the ith unit
        vector.
    :type cell: torch.Tensor
    :param species:
        A list of N strings giving the species of the atoms in the structure, where N is the number of atoms.
    :type species: Sequence[str]
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

    def __init__(self, cell: torch.Tensor, species: Sequence[str], pos: torch.Tensor,
                 property_dict: Optional[dict[str, Any]] = None, metadata: Optional[dict[str, Any]] = None) -> None:
        assert cell.shape == (3, 3)
        assert pos.shape == (len(species), 3)
        self._cell = cell
        self._species = list(species)
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
    def species(self) -> list[str]:
        """
        Return the atom species of the structure.

        :return:
            A list of N strings giving the species of the atoms, where N is the number of atoms.
        :rtype: list[str]
        """
        return self._species

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


class ConfigurationDataset(Dataset):
    """
    A dataset of multiple configurations.
    """
    def __init__(self, configurations: Sequence[Configuration], property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true"):
        super().__init__()
        self.torch_precision = self._get_torch_precision(floating_point_precision)
        self.property_keys = list(property_keys) if property_keys is not None else []
        self._configurations = list(configurations)
        for configuration in self._configurations:
            configuration.to(self.torch_precision)

    @staticmethod
    def _get_torch_precision(floating_point_precision: Union[int, str, None] = "64-true") -> torch.dtype:
        """
        Get the torch precision based on the given floating point precision.

        Args:
            floating_point_precision: Floating point precision to use. Can be one of
                "64-true", "64", "32-true", "32", "16-true", "16", "bf16-true", "bf16",
                "transformer-engine-float16", or "transformer-engine".

        Returns:
            The corresponding torch dtype.
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

    def to_lmdb(self, lmdb_path: Path) -> None:
        """
        Save the configurations to an LMDB file.

        Args:
            lmdb_path: Path to the LMDB file.
        """
        if lmdb_path.exists():
            raise FileExistsError(f"LMDB path {lmdb_path} already exists.")

        with lmdb.Environment(str(lmdb_path), subdir=False, map_size=int(1e12)) as env, env.begin(write=True) as txn:
            for idx, config in tqdm(enumerate(self._configurations), desc=f"Saving configurations to {lmdb_path}",
                                    total=len(self._configurations)):
                data = {
                    "pos": config.pos,
                    "cell": config.cell,
                    "atomic_numbers": [chemical_symbols.index(s) for s in config.species],  # TODO: CORRECT?
                    "ids": config.ids,
                    "pbc": config.pbc,
                } | config.property_dict
                txn.put(str(idx).encode(), pickle.dumps(data))

    def to_ase_atoms(self) -> list[Atoms]:
        """
        Save the configurations to an XYZ file.

        Args:
            xyz_path: Path to the XYZ file.
        """
        all_atoms = []
        for config in tqdm(self._configurations, desc=f"Converting configurations to ASE Atoms"):
            all_atoms.append(Atoms(symbols=config.species, positions=config.pos, cell=config.cell, pbc=config.pbc))
        return all_atoms

    def __len__(self) -> int:
        """
        Get length of the dataset. It is needed to make dataset directly compatible
        with various dataloaders.

        Returns:
            Number of configurations in the dataset.
        """
        return len(self._configurations)

    def __getitem__(self, idx: int) -> Configuration:
        """
        Get the configuration at index `idx`. If the index is a list, it returns a new
        dataset with the configurations at the indices.

        Args:
         idx: Index of the configuration to get or a list of indices.

        Returns:
            The configuration at index `idx` or a new dataset with the configurations at
            the indices.
        """
        return self._configurations[idx]


class LmdbDataset(ConfigurationDataset):
    """

    """

    def __init__(self, lmdb_path: Path, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true") -> None:
        if lmdb_path.exists():
            existing_lmdb_path = lmdb_path
        else:
            # Try to use the data from the omg package.
            existing_lmdb_path = Path(files("omg").joinpath(lmdb_path))
            if not existing_lmdb_path.exists():
                raise FileNotFoundError(f"LMDB path {lmdb_path} neither exists on its own or within the omg package.")

        configs = []
        with (lmdb.Environment(str(existing_lmdb_path), subdir=False, readonly=True, lock=False) as env,
              env.begin() as txn):
            lmdb_configs = [(key.decode(), pickle.loads(data))
                            for key, data in tqdm(txn.cursor(), desc=f"Loading {existing_lmdb_path} data",
                                                  total=txn.stat()["entries"])]
            for key, lmdb_config in lmdb_configs:
                print("Key", key)  # Key just goes from 0 to N-1?
                print("Keys: ", lmdb_config.keys())
                print("ids: ", lmdb_config["ids"])  # That's the fancy stuff
                print("pos ", lmdb_config["pos"])
                if "cell" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'cell'.")
                cell = lmdb_config["cell"]
                if "atomic_numbers" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'atomic_numbers'.")
                # TODO: WHAT THE HELL?
                species = [chemical_symbols[int(i)] for i in lmdb_config["atomic_numbers"]]
                print("species: ", species)
                if "pos" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'pos'.")
                pos = lmdb_config["pos"]
                if "pbc" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'pbc'.")
                if not torch.all(lmdb_config["pbc"]):
                    raise ValueError(f"Configuration with key {key} in the lmdb file does not have full periodic "
                                     f"boundary conditions. OMG only supports configurations with full PBC.")
                if "ids" not in lmdb_config:
                    raise KeyError(f"Key {key} in the lmdb file does not contain 'ids'.")
                ids = lmdb_config["ids"]
                property_dict = {}
                if property_keys is not None:
                    for prop in property_keys:
                        if prop not in lmdb_config:
                            raise KeyError(f"Key {key} in the lmdb file does not contain '{prop}'.")
                        property_dict[prop] = lmdb_config[prop]
                config = Configuration(cell=cell, species=species, pos=pos, ids=ids, property_dict=property_dict,
                                       metadata={"file_path": existing_lmdb_path, "file_key": key})
                configs.append(config)
        super().__init__(configurations=configs, property_keys=property_keys,
                         floating_point_precision=floating_point_precision)


class OverfittingDataset(LmdbDataset):
    """
    Datamodule that always returns a single configuration.

    Can be used to overfit the model to a single configuration.
    """

    def __init__(self, lmdb_path: Path, structure_index: int = 0, property_keys: Optional[Sequence[str]] = None,
                 floating_point_precision: Union[int, str, None] = "64-true") -> None:
        super().__init__(lmdb_path=lmdb_path, property_keys=property_keys,
                         floating_point_precision=floating_point_precision)
        if not 0 <= structure_index < len(self):
            raise KeyError(f"Invalid structure index {structure_index}, "
                           f"possible values are 0 to {len(self) - 1}.")
        self._structure_index = structure_index

    def __getitem__(self, idx: int) -> Configuration:
        """
        Get the configuration at index `idx`. If the index is a list, it returns a new
        dataset with the configurations at the indices.

        This method ignores the given indices and always return the same configuration.

        Args:
         idx: Index of the configuration to get or a list of indices.

        Returns:
            The fixed configuration at a fixed index or a new dataset with the the same configuration replicated.
        """
        return super().__getitem__(self._structure_index)


if __name__ == '__main__':
    dataset = LmdbDataset(lmdb_path=Path("data/mp_20/test.lmdb"))