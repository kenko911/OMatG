from typing import Optional
from ase.data import atomic_numbers
import torch
from torch_geometric.data import Data
from omg.datamodule import Structure
from omg.datamodule.utils import niggli_reduce_configuration, niggli_reduce_data


class OMGData(Data):
    """
    Representation of a single crystalline structure of atoms that is compatible with Pytorch Geometric.

    When using this class with a dataloader from Pytorch Geometric, multiple OMGData objects can be batched together
    into a single batched graph object containing several structures.

    For a batch of size batch_size, the batched graph object will have the following attributes:

    - n_atoms: torch.Tensor of shape (batch_size, ) containing the number of atoms in each structure.
    - species: torch.Tensor of shape (sum(n_atoms), ) containing the atomic numbers of the atoms in the structures.
    - cell: torch.Tensor of shape (batch_size, 3, 3) containing the cell vectors of the structures.
    - batch: torch.Tensor of shape (sum(n_atoms), ) containing the index of the structure to which each atom belongs.
    - pos: torch.Tensor of shape (sum(n_atoms), 3) containing the atomic positions of the atoms in the structures.
    - property: Dictionary containing the properties of the structures.

    Note that Pytorch Geometric dataloaders while batching will initialize the OMGData object without any arguments,
    and then populate the attributes directly. Therefore, the constructor has to handle being called with no arguments.
    In this case, all attributes are initialized to None.

    :param structure:
        Structure object containing the information about the crystalline structure or None.
        Defaults to None
    :type structure: Optional[Structure]
    """

    def __init__(self, structure: Optional[Structure] = None) -> None:
        """Constructor for the OMGData class."""
        super().__init__()
        if structure is None:
            self.n_atoms = None
            self.batch = None
            self.species = None
            self.cell = None
            self.pos = None
            self.property = None
        else:
            self.n_atoms = torch.tensor(len(structure.atomic_numbers))  # Shape: (1, )
            self.batch = torch.zeros(self.n_atoms, dtype=torch.int64)  # Shape: (n_atoms, )
            self.species = structure.atomic_numbers  # Shape: (n_atoms, )
            self.cell = structure.cell.unsqueeze(0)  # Shape: (1, 3, 3).
            self.pos = structure.pos  # Shape: (n_atoms, 3).
            self.property = structure.property_dict

    @classmethod
    def from_omg_configuration(cls, config: Structure, convert_to_fractional=True, niggli=False):  # TODO: REMOVE
        """
        Create a OMGData object from a :class:`omg.datamodule.Configuration` object.

        :param config:  :class:`omg.datamodule.Configuration` object to convert to OMGData
        :param convert_to_fractional: Whether to convert the atomic positions to fractional coordinates
                                    WARNING: This will always convert the atomic positions to fractional coordinates
                                    regardless of the current coordinate system. So, if the atomic positions are already
                                    in fractional coordinates, you need to be careful when setting this flag to True.
        :return:
            OMGData object.
        """
        graph = cls()
        if niggli:
            config = niggli_reduce_configuration(config)

        n_atoms = torch.tensor(len(config.species))
        graph.n_atoms = n_atoms
        graph.batch = torch.zeros(n_atoms, dtype=torch.int64)
        graph.species = torch.tensor([atomic_numbers[z] for z in config.species], dtype=torch.int64)

        assert isinstance(config.cell, torch.Tensor)
        graph.cell = config.cell

        assert isinstance(config.coords, torch.Tensor)
        graph.pos = config.coords

        if config.property_dict is not None:
            graph.property = config.property_dict

        if convert_to_fractional:
            with torch.no_grad():
                graph.pos = torch.remainder(torch.matmul(graph.pos, torch.inverse(graph.cell)), 1.0)

        graph.cell = graph.cell.unsqueeze(0)
        return graph

    @classmethod
    def from_data(cls, species, pos, cell, property_dict={}, convert_to_fractional=True, niggli=False):
        """  # TODO: REMOVE
        Create a OMGData object from the atomic species, positions and cell vectors.

        :param species: Integer array containing the atomic numbers of the atoms
        :param pos: Array containing the atomic positions
        :param cell: Array containing the cell vectors
        :param convert_to_fractional:  Whether to convert the atomic positions to fractional coordinates
                                    WARNING: This will always convert the atomic positions to fractional coordinates
                                    regardless of the current coordinate system. So, if the atomic positions are already
                                    in fractional coordinates, you need to be careful when setting this flag to True.

        :return:
            OMGData object.
        """
        if niggli:
            cell, pos = niggli_reduce_data(species, pos, cell)

        graph = cls()
        n_atoms = torch.tensor(len(species))
        graph.n_atoms = n_atoms
        graph.batch = torch.zeros(n_atoms, dtype=torch.int64)
        if isinstance(species[0], str):
            graph.species = torch.asarray([atomic_numbers[z] for z in species], dtype=torch.int64)
        else:
            graph.species = torch.asarray(species, dtype=torch.int64)

        assert isinstance(cell, torch.Tensor)
        graph.cell = cell

        assert isinstance(pos, torch.Tensor)
        graph.pos = pos

        graph.property = {}
        if convert_to_fractional:
            with torch.no_grad():
                graph.pos = torch.remainder(torch.matmul(graph.pos, torch.inverse(graph.cell).to(graph.pos.dtype)), 1.0)

        graph.property = property_dict

        graph.cell = graph.cell.unsqueeze(0)
        return graph