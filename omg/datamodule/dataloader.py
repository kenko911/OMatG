from ase.data import atomic_numbers
import torch
from torch_geometric.data import Data, Dataset
from omg.datamodule.datamodule import Configuration, ConfigurationDataset
from omg.datamodule.utils import niggli_reduce_configuration, niggli_reduce_data


class OMGData(Data):
    """
    A Pytorch Geometric compatible graph representation of a configuration. When loaded
    into a class:`torch_geometric.data.DataLoader` the graphs of type OMGData
    will be automatically collated and batched.

    OMGData format:
    For a batch size of batch_size, the data format is as follows:
    - n_atoms: torch.Tensor of shape (batch_size, ) containing the number of atoms in each configuration
    - species: torch.Tensor of shape (sum(n_atoms), ) containing the atomic numbers of the atoms in the configurations
    - cell: torch.Tensor of shape (batch_size, 3, 3) containing the cell vectors of the configurations
    - batch: torch.Tensor of shape (sum(n_atoms), ) containing the index of the configuration to which each atom belongs
    - pos: torch.Tensor of shape (sum(n_atoms), 3) containing the atomic positions of the atoms in the configurations
    - property: dict containing the properties of the configurations
    """

    def __init__(self):
        super().__init__()
        self.n_atoms = None
        self.species = None
        self.cell = None
        self.batch = None
        self.pos = None
        self.property = None

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs):
        if "index" in key or "face" in key:
            return self.n_atoms
        elif "batch" in key:
            # number of unique contributions
            return torch.unique(value).size(0)
        else:
            return 0

    def __cat_dim__(self, key: str, value: torch.Tensor, *args, **kwargs):
        if "index" in key or "face" in key:
            return 1
        else:
            return 0

    @classmethod
    def from_omg_configuration(cls, config: Configuration, convert_to_fractional=True, niggli=False):
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
        """
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


class OMGTorchDataset(Dataset):
    """
    This class is a wrapper for the :class:`torch_geometric.data.Dataset` class to enable
    the use of :class:`omg.datamodule.Dataset` as a data source for the graph based models.
    """

    def __init__(self, dataset: ConfigurationDataset, convert_to_fractional=True, niggli=False):
        super().__init__()
        self.dataset = dataset
        self.convert_to_fractional = convert_to_fractional
        self.niggli = niggli

    def len(self):
        return len(self.dataset)

    def get(self, idx: int):
        return OMGData.from_omg_configuration(self.dataset[idx], convert_to_fractional=self.convert_to_fractional,
                                              niggli=self.niggli)
