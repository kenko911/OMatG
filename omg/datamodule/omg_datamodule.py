from typing import Any
from torch_geometric.data.lightning import LightningDataset
from omg.datamodule import OMGDataset, StructureDataset


class OMGDataModule(LightningDataset):
    def __init__(self, train_dataset: StructureDataset, val_dataset: StructureDataset, pred_dataset: StructureDataset,
                 batch_size: int = 1, **kwargs: Any) -> None:
        """
        Lightning DataModule that wraps StructureDataset objects for training, validation, and prediction into
        OMGDataset objects that are compatible with Pytorch Geometric.

        Note that this class extends the LightningDataset class from Pytorch Geometric which in turn extends the
        LightningDataModule class from Pytorch Lightning.

        This class enforces that the datasets used for training, validation, and prediction are all set.

        :param train_dataset:
            StructureDataset for training.
        :type train_dataset: StructureDataset
        :param val_dataset:
            StructureDataset for validation.
        :type val_dataset: StructureDataset
        :param pred_dataset:
            StructureDataset for prediction.
        :type pred_dataset: StructureDataset
        :param batch_size:
            Batch size for data loading. Default is 1.
        :type batch_size: int
        :param kwargs:
            Additional keyword arguments to pass to the LightningDataset constructor which are in turn passed to the
            data loaders.
        :type kwargs: Any
        """
        train_omg_dataset = OMGDataset(train_dataset)
        val_omg_dataset = OMGDataset(val_dataset)
        pred_omg_dataset = OMGDataset(pred_dataset)
        super().__init__(train_dataset=train_omg_dataset, val_dataset=val_omg_dataset, pred_dataset=pred_omg_dataset,
                         batch_size=batch_size, **kwargs)
