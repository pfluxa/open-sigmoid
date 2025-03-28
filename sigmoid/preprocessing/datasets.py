import math
import numpy
import pandas
import h5py
import torch
from torch.utils.data import Dataset as torch_Dataset
from torch.utils.data import IterableDataset as torch_IterableDataset

from sigmoid.preprocessing.coordinate_files import MetaData
from sigmoid.preprocessing.transformations import CategoricalAsOneHot


class StandardDataset(torch_Dataset):

    def __init__(self, path: str, cache_all: bool = False, sort_by: str = None, as_tensor: bool = True):

        super(StandardDataset, self).__init__()

        self.numerical_ = []
        self.categorical_ = []
        self.h5file_ = h5py.File(path, 'r', driver='core') #swmr=True)
        self.sort_idx_ = -1
        self.sort_col_ = None
        self.transform_before_sort_ = False
        self.sort_by_target_ = False
        self.return_tensor_ = as_tensor

        self.x_meta_data_ = MetaData()
        self.y_meta_data_ = MetaData()

        self.x_meta_data_.from_hdf5_attributes(self.h5file_['x'].attrs)
        self.y_meta_data_.from_hdf5_attributes(self.h5file_['y'].attrs)

        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            idx = self.x_meta_data_.get_column_index(col_name)
            if col_type == 'numerical':
                self.numerical_.append(idx)
            if col_type == 'categorical':
                self.categorical_.append(idx)
            if sort_by is None:
                continue
            else:
                if sort_by == col_name:
                    self.sort_idx_ = idx
                    self.sort_col_ = col_name
                if self.x_meta_data_.get_transformed_column_length(col_name) > 1:
                    self.transform_before_sort_ = True

        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            idx = self.y_meta_data_.get_column_index(col_name)
            col_length = self.y_meta_data_.get_transformed_column_length(col_name)
            if col_type == 'numerical':
                self.numerical_.append(idx)
            if col_type == 'categorical':
                self.categorical_.append(idx)
            if sort_by is None:
                continue
            else:
                if sort_by == col_name:
                    self.sort_idx_ = idx
                    self.sort_col_ = col_name
                    self.sort_by_target_ = True
                if self.y_meta_data_.get_transformed_column_length(col_name) > 1:
                    self.transform_before_sort_ = True

        if sort_by is not None and self.sort_idx_ < 0:
            raise RuntimeError(f"Can't sort by column {sort_by} if it does not exists.")

        self.n_rows_ = self.h5file_['x'].shape[0]
        self.n_input_ = self.h5file_['x'].shape[1]

        self.all_cached_ = cache_all
        if self.all_cached_:
            self.x_data_ = numpy.array(self.h5file_['x'][()])
            self.y_data_ = numpy.array(self.h5file_['y'][()])
        if not self.sort_by_target_ and self.sort_idx_ > -1:
            self.x_data_ = self.x_data_[self.x_data_[:, self.sort_idx_].argsort()]
            y = self.y_data_
            if self.transform_before_sort_ and not self.sort_by_target_:
                arg_data = self.x_meta_data_.get_argument_data(self.sort_col_)
                par_data = self.x_meta_data_.get_parameter_data(self.sort_col_)
        elif self.sort_by_target_ and self.sort_idx_ > -1:
            y = self.y_data_
            if self.transform_before_sort_ and self.sort_by_target_:
                arg_data = self.y_meta_data_.get_argument_data(self.sort_col_)
                par_data = self.y_meta_data_.get_parameter_data(self.sort_col_)
                trafo = CategoricalAsOneHot()
                trafo.from_metadata(arg_data, par_data)
                trafo.toggle_direction()
                n = self.y_meta_data_.get_transformed_column_length(self.sort_col_)
                df = pandas.DataFrame(columns=[self.sort_col_ + f'_oh_{x}' for x in range(0, n)], data=y)
                y = trafo(df).to_numpy().reshape((-1, 1))
            self.x_data_ = self.x_data_[y[:, self.sort_idx_].argsort()]
            self.y_data_ = self.y_data_[y[:, self.sort_idx_].argsort()]

    def get_input_dim(self) -> int:
        """ Returns number of features in X data.
        """
        return self.n_input_

    def get_input_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of X.
        """
        return self.x_meta_data_

    def get_output_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of Y.
        """
        return self.y_meta_data_

    def get_input_categorical_columns(self) -> list:
        """ Returns list of columns with ordinal-encoded categories.
        """
        c = [idx for idx in self.categorical_]
        return c

    def get_input_numerical_columns(self) -> list:
        """ Returns list of column ranges with numerical data.
        """
        c = [idx for idx in self.numerical_]
        return c

    def get_input_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.x_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have a 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_output_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights
            for the target variable.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.y_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_split(self, split_type: str, split_id: int, read: bool = True):

        split_idx = None
        if read:
            split_idx = self.h5file_[f"{split_type}_split_{split_id}"][:]
        else:
            split_idx = self.h5file_[f"{split_type}_split_{split_id}"]

        return split_idx


    def get_train_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """

    def get_test_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'test_split_{split_id}']
        if read:
            split = self.h5file_[f'test_split_{split_id}'][:]

        return split

    def get_cardinalities(self):
        """ Returns cardinality of categorical variables.
        """
        n_classes = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'numerical':
                continue
            if col_type == 'categorical':
                params = self.x_meta_data_.get_parameter_data(col_name)
                n_classes.append(len(params['class_names']))

        return n_classes

    def __len__(self):
        """ Returns number of rows in dataset.
        """
        return self.n_rows_

    def __getitem__(self, index):
        """ Returns X in a "transformer-friendly" format.
        """
        x_data = None
        y_data = None
        if self.all_cached_:
            x_data = self.x_data_[index, :]
            y_data = self.y_data_[index, :]
        else:
            x_data = self.h5file_['x'][index, :]
            y_data = self.h5file_['y'][index, :]

        if self.return_tensor_:
            X = torch.tensor(x_data, dtype=torch.float32)
            y = torch.tensor(y_data, dtype=torch.float32)
        else:
            X = numpy.atleast_1d(x_data)
            y = numpy.atleast_1d(y_data)

        return X, y


class GroupedDataset(torch_Dataset):

    def __init__(self, path: str, column_groups: list, cache_all: bool = False):

        super(GroupedDataset, self).__init__()

        self.numerical_ = []
        self.categorical_ = []
        self.grouped_input_idx_ = []
        for _ in range(len(column_groups)):
            self.grouped_input_idx_.append([])
        self.grouped_target_idx_ = []
        for _ in range(len(column_groups)):
            self.grouped_target_idx_.append([])
        self.h5file_ = h5py.File(path, 'r', driver='core') #swmr=True)
        self.sort_idx_ = -1
        self.sort_col_ = None
        self.transform_before_sort_ = False
        self.sort_by_target_ = False

        self.x_meta_data_ = MetaData()

        self.x_meta_data_.from_hdf5_attributes(self.h5file_['x'].attrs)

        offset = 0
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            idx = self.x_meta_data_.get_column_index(col_name)
            col_length = self.x_meta_data_.get_transformed_column_length(col_name)
            if col_type == 'numerical':
                self.numerical_.append(idx)
            if col_type == 'categorical':
                self.categorical_.append(idx)
            for i, col_group in enumerate(column_groups):
                for col in col_group['inputs']:
                    if col == col_name:
                        for j in range(0, col_length):
                            self.grouped_input_idx_[i].append(idx + j + offset)
                        offset += col_length - 1
                for col in col_group['targets']:
                    if col == col_name:
                        for j in range(0, col_length):
                            self.grouped_target_idx_[i].append(idx + j + offset)
                        offset += col_length - 1

        self.n_rows_ = self.h5file_['x'].shape[0]
        self.n_input_ = self.h5file_['x'].shape[1]

        self.all_cached_ = cache_all
        if self.all_cached_:
            self.x_data_ = numpy.array(self.h5file_['x'][()])


    def get_input_dim(self) -> int:
        """ Returns number of features in X data.
        """
        return self.n_input_

    def get_input_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of X.
        """
        return self.x_meta_data_

    def get_output_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of Y.
        """
        return self.y_meta_data_

    def get_input_categorical_columns(self) -> list:
        """ Returns list of columns with ordinal-encoded categories.
        """
        c = [idx for idx in self.categorical_]
        return c

    def get_input_numerical_columns(self) -> list:
        """ Returns list of column ranges with numerical data.
        """
        return self.numerical_

    def get_input_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.x_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have a 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_output_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights
            for the target variable.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.y_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_train_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'train_split_{split_id}']
        if read:
            split = self.h5file_[f'train_split_{split_id}'][:]

        return split

    def get_test_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'test_split_{split_id}']
        if read:
            split = self.h5file_[f'test_split_{split_id}'][:]

        return split

    def get_cardinalities(self):
        """ Returns cardinality of categorical variables.
        """
        n_classes = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'numerical':
                continue
            if col_type == 'categorical':
                params = self.x_meta_data_.get_parameter_data(col_name)
                n_classes.append(len(params['class_names']))

        return n_classes

    def __len__(self):
        """ Returns number of rows in dataset.
        """
        return self.n_rows_

    def __getitem__(self, index):
        """ Returns X in a "transformer-friendly" format.
        """
        grouped = []
        for grp_idx_input, grp_idx_target in zip(self.grouped_input_idx_, self.grouped_target_idx_):
            x_data = None
            if self.all_cached_:
                x_data = self.x_data_[index, grp_idx_input]
                y_data = self.x_data_[index, grp_idx_target]
            else:
                x_data = self.h5file_['x'][index, grp_idx_input]
                y_data = self.h5file_['x'][index, grp_idx_target]
            X = torch.tensor(x_data, dtype=torch.float32)
            y = torch.tensor(y_data, dtype=torch.float32)
            grouped.append((X, y))

        return grouped


class BatchMixedTypesAutoencoderDataset(torch_IterableDataset):
    """ Wrapper around torch.utils.data.IterableDataset to read Cache files.

        Allowed types:
            - numerical
            - categorical (as numbers, i.e. the output of a label encoder)
    """
    def __init__(self, path: str, cache_all: bool = False) -> None:
        """ Initializer for Dataset.

            @param path: str
                system path to HDF5 file with data.
        """
        super(BatchMixedTypesAutoencoderDataset, self).__init__()

        self.numerical_ = []
        self.categorical_ = []
        # swmr = True allows multiple workers to read from the same file
        # concurrently. handy!
        self.h5file_ = h5py.File(path, 'r', swmr=True)
        # use meta data to tell which columns are for training and
        # which columns are for target
        self.x_meta_data_ = MetaData()
        self.y_meta_data_ = MetaData()

        self.x_meta_data_.from_hdf5_attributes(self.h5file_['x'].attrs)
        self.y_meta_data_.from_hdf5_attributes(self.h5file_['y'].attrs)

        # count number of numerical columns
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            idx = self.x_meta_data_.get_column_index(col_name)
            if col_type == 'numerical':
                self.numerical_.append(idx)
            if col_type == 'categorical':
                self.categorical_.append(idx)

        self.n_rows_ = self.h5file_['x'].shape[0]
        self.n_input_ = self.h5file_['x'].shape[1]

        # self.all_cached_ = True
        # self.x_data = self.h5file_['x'][()]
        # self.y_data = self.h5file_['y'][()]
        self.shuffle_ = False
        self.idx_ = None
        self.return_y_ = False

        self.start = 0
        self.end = self.n_rows_
        self.rng_ = numpy.random.default_rng(seed=42)

    def get_input_dim(self) -> int:
        """ Returns number of features in X data.
        """
        return self.n_input_

    def get_input_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of X.
        """
        return self.x_meta_data_

    def get_output_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of Y.
        """
        return self.y_meta_data_

    def get_input_categorical_columns(self) -> list:
        """ Returns list of columns with ordinal-encoded categories.
        """
        c = [idx for idx in self.categorical_]
        return c

    def get_input_numerical_columns(self) -> list:
        """ Returns list of column ranges with numerical data.
        """
        n = [idx for idx in self.numerical_]
        return n

    def get_input_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.x_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have a 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_y_data_types(self):

        col_types = []
        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            col_types.append(col_type)

        return col_types

    def get_output_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights
            for the target variable.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.y_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def set_as_dataset(self):

        self.x_data = self.h5file_['x'][()]
        self.y_data = self.h5file_['y'][()]

    def set_as_training(self, split: int):

        idx = self.get_train_split(split_id=split, read=True)
        idx = numpy.sort(idx)
        self.x_data = self.h5file_['x'][idx, :]
        self.y_data = self.h5file_['y'][idx, :]

        self.start = 0
        self.end = len(idx)

    def set_as_testing(self, split: int):

        idx = self.get_test_split(split_id=split, read=True)
        idx = numpy.sort(idx)
        self.x_data = self.h5file_['x'][idx, :]
        self.y_data = self.h5file_['y'][idx, :]

        self.start = 0
        self.end = len(idx)

    def get_train_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'train_split_{split_id}']
        if read:
            split_data = self.h5file_[f'train_split_{split_id}'][()]
            return split_data

        return split

    def get_test_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'test_split_{split_id}']
        if read:
            split_data = self.h5file_[f'test_split_{split_id}'][()]
            return split_data

        return split

    def get_cardinalities(self):
        """ Returns cardinality of categorical variables.
        """
        n_classes = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                params = self.x_meta_data_.get_parameter_data(col_name)
                n_classes.append(len(params['class_names']))

        return n_classes

    def toggle_return_y(self):

        self.return_y_ = not self.return_y_

    def shuffle(self, idx = None):

        if idx is None:
            idx = numpy.arange(self.start, self.end, 1)
            self.rng_.shuffle(idx)
        self.x_data = self.x_data[idx]
        self.y_data = self.y_data[idx]

        return idx

    def __len__(self):
        """ Returns number of rows in dataset.
        """
        if self.end % 256 == 0:
            return self.end // 256
        return self.end // 256 + 1

    def __iter__(self):
        """ Returns X in a "transformer-friendly" format.
        """
        iter_start = self.start
        iter_end = self.end
        worker_info = torch.utils.data.get_worker_info()
        # single-process data loading, return the full iterator
        if worker_info is None:  # single-process data loading, return the full iterator
            pass
        # in a worker process
        else:
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        idx = numpy.arange(iter_start, iter_end, 1)
        # self.rng_.shuffle(idx)
        x_data = self.x_data[idx]
        x_cat = x_data[:, self.categorical_].astype(numpy.int64)
        x_num = x_data[:, self.numerical_].astype(numpy.float32)
        if self.return_y_:
            y_data = self.y_data[idx]

        bs = 256
        for offset in range(0, iter_end - iter_start, bs):
            x_num_t = torch.from_numpy(x_num[offset:offset + bs, :])
            x_cat_t = torch.from_numpy(x_cat[offset:offset + bs, :])
            if self.return_y_:
                y = y_data[offset:offset + bs, :]
                for y_type in self.get_y_data_types():
                    if y_type == 'categorical':
                        y = y.astype(numpy.int64)
                    else:
                        y = y.astype(numpy.float32)
                y_t = torch.from_numpy(y)
                yield (x_num_t, x_cat_t, y_t)
            else:
                yield (x_num_t, x_cat_t)


class MixedTypesAutoencoderDataset(torch_Dataset):
    """ Wrapper around torch.utils.data.Dataset to read Cache files.

        This particular object handles LocalCache objects, i.e.,
        works for non-distributed environments.

        `LocalDataset` can be thought as a DataFrame with some added
        functionality to keep track of column types. This is needed
        by `sigmoid` in the essenziehen step. The most important
        functionality is that it "knows" which columns correspond
        to a given type. Supported types are

        - numerical
        - categorical (as numers i.e. the output of a label encoder)
    """
    def __init__(self, path: str, cache_all: bool = False) -> None:
        """ Initializer for `LocalDataset`.

            @param path: str
                system path to HDF5 file with data.
        """
        super(MixedTypesAutoencoderDataset, self).__init__()

        self.numerical_ = []
        self.categorical_ = []
        self.cardinalities_ = []
        # swmr = True allows multiple workers to read from the same file
        # concurrently. handy!
        self.h5file_ = h5py.File(path, 'r', driver='core') #swmr=True)
        # use meta data to tell which columns are for training and
        # which columns are for target
        self.x_meta_data_ = MetaData()
        self.y_meta_data_ = MetaData()

        self.x_meta_data_.from_hdf5_attributes(self.h5file_['x'].attrs)
        self.y_meta_data_.from_hdf5_attributes(self.h5file_['y'].attrs)

        # count number of numerical columns
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            idx = self.x_meta_data_.get_column_index(col_name)
            if col_type == 'numerical':
                self.numerical_.append(idx)
            if col_type == 'categorical':
                params = self.x_meta_data_.get_parameter_data(col_name)
                n_classes = len(params['class_names'])
                self.categorical_.append(idx)
                self.cardinalities_.append(n_classes)

        self.n_rows_ = self.h5file_['x'].shape[0]
        self.n_input_ = self.h5file_['x'].shape[1]

        self.all_cached_ = cache_all
        if self.all_cached_:
            self.x_data = numpy.array(self.h5file_['x'][:])
            self.y_data = numpy.array(self.h5file_['y'][:])

        self.return_y_ = False

    def get_split(self, split_type: str, split_id: int, read: bool = True):

        split_idx = None
        if read:
            split_idx = self.h5file_[f"{split_type}_split_{split_id}"][:]
        else:
            split_idx = self.h5file_[f"{split_type}_split_{split_id}"]

        return split_idx

    def get_column_min_max(self, column_idx: int) -> tuple:
        """ Returns minimum and maximum values for a given column.
        """
        if self.all_cached_:
            min_val = self.x_data[:, column_idx].min()
            max_val = self.x_data[:, column_idx].max()
        else:
            min_val = self.h5file_['x'][:, column_idx].min()
            max_val = self.h5file_['x'][:, column_idx].max()

        return min_val, max_val

    def get_column_nunique_values(self, column_idx: int) ->int:
        """ Returns unique values for a given column.
        """
        if self.all_cached_:
            unique_values = numpy.unique(self.x_data[:, column_idx])
        else:
            unique_values = numpy.unique(self.h5file_['x'][:, column_idx])

        return len(unique_values.tolist())

    def get_input_dim(self) -> int:
        """ Returns number of features in X data.
        """
        return self.n_input_

    def get_input_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of X.
        """
        return self.x_meta_data_

    def get_output_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of Y.
        """
        return self.y_meta_data_

    def get_input_categorical_columns(self) -> list:
        """ Returns list of columns with ordinal-encoded categories.
        """
        c = [idx for idx in self.categorical_]
        return c

    def get_input_numerical_columns(self) -> list:
        """ Returns list of column ranges with numerical data.
        """
        c = [idx for idx in self.numerical_]
        return c

    def get_input_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.x_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have a 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_output_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights
            for the target variable.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.y_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation does not have 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_train_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'train_split_{split_id}']
        if read:
            split = self.h5file_[f'train_split_{split_id}'][:]

        return split

    def get_test_split(self, split_id: int = 0, read: bool = True) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'test_split_{split_id}']
        if read:
            split = self.h5file_[f'test_split_{split_id}'][:]

        return split

    def get_cardinalities(self):
        """ Returns cardinality of categorical variables.
        """
        n_classes = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'numerical':
                continue
            if col_type == 'categorical':
                params = self.x_meta_data_.get_parameter_data(col_name)
                n_classes.append(len(params['class_names']))

        return n_classes

    def toggle_return_y(self):

        self.return_y_ = not self.return_y_

    def get_type_features(self, column_type: str):

        data = []
        col_names = []
        x_data = None
        if self.all_cached_:
            x_data = self.x_data[:, :]
        else:
            x_data = self.h5file_['x'][:, :]

        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            idx = self.x_meta_data_.get_column_index(col_name)
            if col_type == column_type:
                col_names.append(col_name)
                data.append(x_data[:, None, idx])

        return col_names, numpy.concatenate(data, axis=1)

    def __len__(self):
        """ Returns number of rows in dataset.
        """
        return self.n_rows_

    def __getitem__(self, index):
        """ Returns X in a "transformer-friendly" format.
        """
        x_data = None
        if self.all_cached_:
            x_data = self.x_data[index, :]
        else:
            x_data = self.h5file_['x'][index, :]

        y_data = None
        if self.return_y_:
            if self.all_cached_:
                y_data = self.y_data[index, :]
            else:
                y_data = self.h5file_['y'][index, :]

        x_cat = torch.tensor(x_data[self.categorical_]).long()
        x_num = torch.tensor(x_data[self.numerical_]).float()

        if self.return_y_:
            y_data = torch.tensor(y_data).float()
            return (x_num, x_cat, y_data)

        return (x_num, x_cat)


class MixedTypesDataset_old(torch_Dataset):
    """ Wrapper around torch.utils.data.Dataset to read Cache files.

        This particular object handles LocalCache objects, i.e.,
        works for non-distributed environments.

        `LocalDataset` can be thought as a DataFrame with some added
        functionality to keep track of column types. This is needed
        by `sigmoid` in the essenziehen step. The most important
        functionality is that it "knows" which columns correspond
        to a given type. Supported types are

        - numerical
        - categorical (as one-hot encoded vectors)
        - binary (zero and ones)
    """
    def __init__(self, path: str) -> None:
        """ Initializer for `LocalDataset`.

            @param path: str
                system path to HDF5 file with data.
        """
        self.binary_ = []
        self.categorical_ = []
        self.numerical_ = []

        # swmr = True allows multiple workers to read from the same file
        # concurrently. handy!
        self.h5file_ = h5py.File(path, 'r', driver='core') #swmr=True)
        # use meta data to tell which columns are for training and
        # which columns are for target
        self.x_meta_data_ = MetaData()
        self.y_meta_data_ = MetaData()

        self.x_meta_data_.from_hdf5_attributes(self.h5file_['x'].attrs)
        self.y_meta_data_.from_hdf5_attributes(self.h5file_['y'].attrs)

        # assemble column ranges for different input types
        offset = 0
        for col_name in self.x_meta_data_.get_columns():
            idx = self.x_meta_data_.get_column_index(col_name)
            # save ranges of different column types
            col_type = self.x_meta_data_.get_column_type(col_name)
            size = 0
            if col_type in ['numerical', 'binary']:
                size = 1
            elif col_type == 'categorical':
                size = self.x_meta_data_.get_transformed_column_length(col_name)
            col_range = numpy.arange(idx + offset,
                                     idx + offset + size,
                                     dtype=int)
            col_range = col_range.tolist()
            if col_type == 'numerical':
                self.numerical_.append(col_range)
            elif col_type == 'categorical':
                self.categorical_.append(col_range)
            elif col_type == 'binary':
                self.binary_.append(col_range)
            offset += size - 1

        self.x_data_ = numpy.asarray(self.h5file_['x'][:])
        self.y_data_ = numpy.asarray(self.h5file_['y'][:])

        self.n_rows_ = self.x_data_.shape[0]
        self.n_input_ = self.x_data_.shape[1]
        self.n_output_ = self.y_data_.shape[1]

        self.x_num_data_ = None
        self.x_cat_data_ = None
        self.x_bin_data_ = None


    def preprocess(self):
        """ Separates data types into numerical, categorical and binary.
        """
        x_num = []
        new_num_col_ranges = []
        x_cat = []
        new_cat_col_ranges = []
        x_bin = []
        new_bin_col_ranges = []
        # numerical columns
        offset = 0
        for col_range in self.numerical_:
            x_num.append(self.x_data_[:, col_range])
            size = 1
            new_col_range = numpy.arange(
                                offset,
                                offset + size,
                                dtype=int)
            offset += size
            new_num_col_ranges.append(new_col_range.tolist())
        # categorical columns
        offset = 0
        for col_range in self.categorical_:
            x_cat.append(self.x_data_[:, col_range])
            size = col_range[1] - col_range[0]
            new_col_range = numpy.arange(
                                offset,
                                offset + size + 1,
                                dtype=int)
            offset += size + 1
            new_cat_col_ranges.append(new_col_range.tolist())
        # binary columns
        offset = 0
        for col_range in self.binary_:
            x_bin.append(self.x_data_[:, col_range])
            size = 1
            new_col_range = numpy.arange(
                                offset,
                                offset + size,
                                dtype=int)
            offset += size
            new_bin_col_ranges.append(new_col_range.tolist())

        self.numerical_ = new_num_col_ranges
        if len(x_num) == 1:
            self.x_num_data_ = x_num[0]
        elif len(x_num) > 1:
            self.x_num_data_ = numpy.concatenate(x_num, axis=1)

        self.categorical_ = new_cat_col_ranges
        if len(x_cat) == 1:
            self.x_cat_data_ = x_cat[0]
        elif len(x_cat) > 1:
            self.x_cat_data_ = numpy.concatenate(x_cat, axis=1)

        self.binary_ = new_bin_col_ranges
        if len(x_bin) == 1:
            self.x_bin_data_ = x_bin[0]
        elif len(x_bin) > 1:
            self.x_bin_data_ = numpy.concatenate(x_bin, axis=1)

    def get_input_dim(self) -> int:
        """ Returns number of features in X data.
        """
        return self.n_input_

    def get_output_dim(self) -> int:
        """ Returns number of features in Y data.
        """
        return self.n_output_

    def get_input_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of X.
        """
        return self.x_meta_data_

    def get_output_metadata(self) -> MetaData:
        """ Returns reference to internal metadata of Y.
        """
        return self.y_meta_data_

    def get_input_numerical_columns(self) -> list:
        """ Returns list of column ranges with numerical data.
        """
        return self.numerical_

    def get_input_categorical_columns(self) -> list:
        """ Returns list of column ranges with one-hot encoded data.
        """
        return self.categorical_

    def get_input_binary_columns(self) -> list:
        """ Returns list of column ranges with binary data.
        """
        return self.binary_

    def get_input_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.x_meta_data_.get_columns():
            col_type = self.x_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.x_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation needs 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_output_class_weights(self) -> list:
        """ Returns list of numpy.ndarray with relative class weights
            for the target variable.

            Weights are calculated as normalized relative frequencies
            of occurrences. It is assumed that the transformation that
            encoded categories has a 'class_weights' parameter.

            @raises: ValueError if transformation does not have a
            parameter called 'class_weights'.
        """
        weights = []
        for col_name in self.y_meta_data_.get_columns():
            col_type = self.y_meta_data_.get_column_type(col_name)
            if col_type == 'categorical':
                p = self.y_meta_data_.get_parameter_data(col_name)
                if 'class_weights' not in p:
                    raise ValueError(
                        "Transformation needs 'class_weights' parameter.")
                w = p['class_weights']
                w = numpy.asarray(w).ravel()
                w = w / numpy.sum(w)
                weights.append(w)

        return weights

    def get_train_split(self, split_id: int = 0) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'train_split_{split_id}'][:]

        return split

    def get_test_split(self, split_id: int = 0) -> numpy.ndarray:
        """ Returns training split
        """
        split = self.h5file_[f'test_split_{split_id}'][:]

        return split

    def __len__(self):
        """ Returns number of rows in dataset.
        """
        return self.n_rows_

    def __getitem__(self, index):
        """ Returns tensors X and Y.
        """
        X_num = torch.zeros((1,), dtype=torch.float32)
        if len(self.numerical_) > 0:
            x_num = self.x_num_data_[index]
            x_num = numpy.atleast_1d(x_num)
            X_num = torch.tensor(x_num, dtype=torch.float32)

        X_cat = torch.zeros((1,), dtype=torch.float32)
        if len(self.categorical_) > 0:
            x_cat = self.x_cat_data_[index]
            x_cat = numpy.atleast_1d(x_cat)
            X_cat = torch.tensor(x_cat, dtype=torch.float32)

        X_bin = torch.zeros((1,), dtype=torch.float32)
        if len(self.binary_) > 0:
            x_bin = self.x_bin_data_[index]
            x_bin = numpy.atleast_1d(x_bin)
            X_bin = torch.tensor(x_bin, dtype=torch.float32)

        y = numpy.atleast_1d(self.y_data_[index])
        Y = torch.tensor(y, dtype=torch.float32)

        return (X_num, X_cat, X_bin), Y
