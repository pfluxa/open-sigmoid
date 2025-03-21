import math

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader as torch_DataLoader

from sigmoid.preprocessing.datasets import torch_Dataset

@staticmethod
def collate_fn(batch):
    
    data = batch[0]
    x_num = data[0]
    x_cat = data[1]
    if len(data) == 3:
        y = data[2]
        return (x_num, x_cat, y)
     
    return (x_num, x_cat)


class MultiprocessLoader(torch_DataLoader):
    """ Wrapper around torch.utils.data.DataLoader with
        support for distributed data loading.
    """

    def __init__(self, dataset, train_dataset, test_dataset) -> None:
        """ Initializer.

            @param dataset: LocalDataset
                An instance of LocalDataset.
        """
        super(MultiprocessLoader, self).__init__(torch_DataLoader)

        self.dataset_ = dataset
        self.train_dataset_ = train_dataset
        self.test_dataset_ = test_dataset

    def get_dataset_loader(self, num_workers: int = 0):
        """ Returns DataLoader that iterates over Dataset
            in order, that is, following the original index.
        """
        loader = None
        if num_workers > 0 and num_workers is not None:
            loader = torch_DataLoader(
                self.dataset_,
                num_workers=num_workers,
                persistent_workers=True,
                drop_last=False,
                collate_fn=collate_fn)
        else:
            loader = torch_DataLoader(
                self.dataset_,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=collate_fn)

        return loader
        

    def get_train_loader(self,
                         split_id: int,
                         shuffle: bool = True,
                         num_workers: int = 0):
        """ Returns DataLoader for training.

            @param split_id: integer
                id of split to load.
            @param batch_size: integer
                batch size, passed to `batch_size` when
                building the `DataLoader`
            @param num_workers: integer (defaults to 0)
                number of workers, passed as to `num_workers`
                when building the `DataLoader`
        """
        loader = None
        if num_workers > 0 and num_workers is not None:
            loader = torch_DataLoader(
                self.train_dataset_,
                num_workers=num_workers,
                persistent_workers=True,
                drop_last=False,
                collate_fn=collate_fn)
        else:
            loader = torch_DataLoader(
                self.train_dataset_,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=collate_fn)
            
        return loader

    def get_test_loader(self,
                        split_id: int,
                        shuffle: bool = False,
                        num_workers: int = 0):
        """ Returns DataLoader for training.

            @param split_id: integer
                id of split to load.
            @param batch_size: integer
                batch size, passed to `batch_size` when building
                the `DataLoader`.
            @param num_workers: integer (defaults to 0)
                number of workers, passed as to `num_workers` when
                building the `DataLoader`.
        """
        loader = None
        if num_workers > 0 and num_workers is not None:
            loader = torch_DataLoader(
                self.test_dataset_,
                num_workers=num_workers,
                persistent_workers=True,
                drop_last=False,
                collate_fn=collate_fn)
        else:
            loader = torch_DataLoader(
                self.test_dataset_,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=collate_fn)
            
        return loader

class StandardLoader(torch_DataLoader):
    """ Wrapper around torch.utils.data.DataLoader with
        support for distributed data loading.
    """

    def __init__(self, dataset) -> None:
        """ Initializer.

            @param dataset: LocalDataset
                An instance of LocalDataset.
        """
        super(StandardLoader, self).__init__(dataset)
        self.dataset_ = dataset

    def get_loader(self, split_type, split_id, batch_size,
                   num_workers: int = 0,
                   shuffle: bool = True):
        split_dataset = self.dataset_
        if split_type != "all":
            split_idx = self.dataset_.get_split(split_type, split_id, read=True)
            split_dataset = Subset(self.dataset_, split_idx.tolist())
        
        loader = torch_DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False)

        return loader
        
    def get_train_loader(self,
                         split_id: int,
                         batch_size: int, 
                         shuffle: bool = True,
                         num_workers: int = 0):
        """ Returns DataLoader for training.

            @param split_id: integer
                id of split to load.
            @param batch_size: integer
                batch size, passed to `batch_size` when
                building the `DataLoader`
            @param num_workers: integer (defaults to 0)
                number of workers, passed as to `num_workers`
                when building the `DataLoader`
        """
        train_split = self.dataset_.get_train_split(split_id).tolist()
        train_dataset = Subset(self.dataset_, train_split)

        if num_workers > 0 and num_workers is not None:
            loader = torch_DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True)
        else:
            loader = torch_DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=True)

        return loader

    def get_test_loader(self,
                        split_id: int,
                        batch_size: int, 
                        shuffle: bool = False,
                        num_workers: int = 0):
        """ Returns DataLoader for training.

            @param split_id: integer
                id of split to load.
            @param batch_size: integer
                batch size, passed to `batch_size` when building
                the `DataLoader`.
            @param num_workers: integer (defaults to 0)
                number of workers, passed as to `num_workers` when
                building the `DataLoader`.
        """
        test_split = self.dataset_.get_test_split(split_id).tolist()
        test_dataset = Subset(self.dataset_, test_split)

        if num_workers > 0 and num_workers is not None:
            loader = torch_DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False)
        else:
            loader = torch_DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=False)

        return loader
