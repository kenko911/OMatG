from typing import Any, Optional, Sequence
from ase import Atoms
from ase.symbols import Symbols
from pymatgen.core import Structure as PymatgenStructure, Lattice as PymatgenLattice
import torch


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
