"""
coordinate_files.py

File containing the implementation of data cache and dataset.

A cache is an abstraction that enables data to be stored
in binary format so as to be read efficiently by data loaders
(e.g. PyTorch DataLoader) in both local and distributed environments.
It basically serves the same purpose of a DataFrame, with the added
capabililty of performing and storing data transformations and
information about the data types. As such, cache objects contain
a purely numerical representation of the data, and hold information
about what is being stored, in particular, the type of data and the
way it is presented to the data loader.
"""
import os
import copy
import json
from collections import OrderedDict

import h5py
import numpy
import pandas

import torch

from torch.masked import masked_tensor, as_masked_tensor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from sigmoid.preprocessing.cleaners import ColumnCleaner
from sigmoid.preprocessing.transformations import ColumnTransform, CategoricalAsOneHot


class MetaData(OrderedDict):
    """ Object to handle column metadata.

    Implementation of MetaData object for SIGMOID.

    MetaData objects are a critical component of data handling: they
    store column information in such a way that it can be read/written
    to/fropm HDF5 files, which is the default format to load data into
    the machine learning pipeline of SIGMOID.

    MetaData object contain several key attributes:

        - original index of column (as in the raw dataframe)
        - type of column (numerical, categorical, etc)
        - length of the transformed column (e.g. for one-hot encoding)
        - id of associated transformation
        - transformation arguments and parameters

    This information is represented as numpy.ndarrays which allows for
    persistent storage within the HDF5 file itself, making the Cache
    (see coordinate_files.py) a self-contained representation of the data.

    Metadata is stored using the attributes capability of HDF5
    files, and as such it must be stored as vectorized representation.
    This implementation takes care of "translating" this vectorized
    format to a "human readable" representation.

    MetaData inherits from OrderedDict.
    """
    def __init__(self, max_buffer_size: int = 4096 * 4):
        """ Initializer of meta data object.

            @param max_buffer_size: int (defaults to 100)
                maximum *combined* number of arguments and parameters
                used by a column transformation routine.
        """
        super(MetaData, self).__init__()
        self.buffer_size_ = max_buffer_size
        self.pos_col_index_ = 0
        self.pos_col_type_ = 1
        self.pos_trafo_col_size_ = 2
        self.pos_trafo_n_params_ = 3
        self.pos_trafo_n_args_ = 4
        self.pos_begin_data_ = 10

    def from_hdf5_attributes(self, attr):
        """ Loads metadata information from h5py attribute object.

            @param attr: h5py.File.attr instance.
        """
        self.update(attr)

    def to_dict(self) -> OrderedDict:
        """ Returns deep-copy of object as OrderedDict.
        """
        return copy.deepcopy(self)

    def get_columns(self):
        """ Returns available column names.
        """
        return list(self.keys())

    def add_column(self,
                   col_name: str, col_index: int, col_type: str,
                   col_transform=None):
        """ Adds column information to metadata object.

            @param col_name: str
                name of the column being added
            @param col_index: int
                position of column in the original representation,
                usually a pandas.DataFrame.
            @param col_type: str
                type of column. Valid types are 'float', 'integer',
                'categorical', 'binary' and 'datetime'.
            @param col_transform: ColumnTransformation (defaults to None)
                instance of a ColumnTransformation object
        """
        # arguments passed to column transformation
        trafo_args = {}
        # parameters passed to column transformation
        trafo_params = {}
        # size of transformed column
        trafo_size = 1
        if col_transform is not None:
            # transformation length
            trafo_size = col_transform.get_length()
            # transformation parameters as numpy.ndarray
            trafo_params = col_transform.get_parameters()
            # transformation arguments as numpy.ndarray
            trafo_args = col_transform.get_arguments()

        # buffer to store column information
        data = [''] * self.buffer_size_
        data[self.pos_col_index_] = str(col_index)
        data[self.pos_col_type_] = str(col_type)
        data[self.pos_trafo_col_size_] = str(trafo_size)
        data[self.pos_trafo_n_params_] = str(len(trafo_params))
        data[self.pos_trafo_n_args_] = str(len(trafo_args))

        # store transformation arguments
        offset = self.pos_begin_data_
        offset += 1
        for arg_name, arg_data in trafo_args.items():
            data[offset] = str(len(arg_name))
            offset += 1
            data[offset] = str(len(arg_data))
            offset += 1
            for c in arg_name:
                data[offset] = str(c)
                offset += 1
            for d in arg_data:
                if type(d) == str:
                    data[offset] = d
                else:
                    data[offset] = "{:.6f}".format(float(arg_data))
                offset += 1
        # store transformation parameters
        for param_name, param_data in trafo_params.items():
            param_size = len(param_data)
            data[offset] = str(len(param_name))
            offset += 1
            data[offset] = str(len(param_data))
            offset += 1
            for c in param_name:
                data[offset] = str(c)
                offset += 1
            for i in range(param_size):
                data[offset] = str(param_data[i])
                offset += 1
        # build buffer
        bytebuffer = ''
        for d in data:
            bytebuffer += d + '\t'
        # consolidate into OrderedDict as byte-buffer
        self[col_name] = bytebuffer  #.encode()

    def get_column_index(self, col_name: str):
        """ Returns index of column.
        """
        r = self[col_name].split('\t')[self.pos_col_index_]
        return int(r)

    def get_column_type(self, col_name: str):
        """ Returns column type as string.
        """
        data = self[col_name].split('\t')
        r = data[self.pos_col_type_]
        return r

    def get_transformed_column_length(self, col_name: str):
        """ Returns column "length".

            Here, length is used to refer to the number of
            elements in a column. For instance, a categorical
            column that is encoded has a one-hot vector and has
            n categories will have a length of n.
        """
        r = self[col_name].split('\t')[self.pos_trafo_col_size_]
        return int(r)

    def get_parameter_data(self, col_name: str, return_offset: bool = False) -> dict:
        """ Return list with transformation parameter.
            @param col_name: str
                name of the column
            @param return_offset: optional, bool
                whether to return the offset in the buffer.
        """
        data = self[col_name].split('\t')
        n_param = int(data[self.pos_trafo_n_params_])
        # get offset to load parameter data
        args, offset = self.get_argument_data(col_name, return_offset=True)
        # reconstruct parameters
        parameters = {}
        for _ in range(0, n_param):
            param_name_len = int(data[offset])
            offset += 1
            param_size = int(data[offset])
            offset += 1

            param_chars = []
            for c in data[offset:offset + param_name_len]:
                param_chars.append(c)
                offset += 1
            param_name = ''.join(param_chars)

            param_data = []
            for __ in range(param_size):
                r = None
                try:
                    r = float(data[offset])
                except ValueError:
                    r = data[offset]
                param_data.append(r)
                offset += 1
            parameters[param_name] = param_data

        if return_offset:
            return parameters, offset
        return parameters

    def get_argument_data(self, col_name: str, return_offset: bool = False) -> dict:
        """ Returns list with transformation arguments.
        """
        data = self[col_name].split('\t')
        offset = self.pos_begin_data_
        n_args = int(data[self.pos_trafo_n_args_])

        offset += 1
        arguments = {}
        for _ in range(0, n_args):
            arg_name_len = int(data[offset])
            offset += 1
            arg_size = int(data[offset])
            offset += 1

            arg_chars = []
            for c in data[offset:offset + arg_name_len]:
                arg_chars.append(c)
                offset += 1
            arg_name = ''.join(arg_chars)

            arg_data = []
            for __ in range(arg_size):
                r = None
                try:
                    r = float(data[offset])
                except ValueError:
                    r = data[offset]
                arg_data.append(r)
                offset += 1
            arguments[arg_name] = arg_data

        if return_offset:
            return arguments, offset

        return arguments


class Cache:
    """
    Cache object for non-distributed environments.
    """

    def __init__(self, dataframe: pandas.DataFrame,
                 target: str = None,
                 ignore: list = []) -> None:
        """ Initializes cache object.

        @param dataframe: pandas.DataFrame
            instance of pandas.DataFrame with data.
        @param target: str
            name of target column
        @param ignore: list (defalts to empty)
            list of column names from dataframe that must be ignored
        """
        self.target_ = target
        self.ignore_ = ignore
        self.data_ = dataframe.drop(ignore, axis=1)
        self.x_data_ = pandas.DataFrame()
        self.y_data_ = pandas.DataFrame()

        self.n_rows = dataframe.shape[0]

        self.x_meta_data_ = MetaData()
        self.y_meta_data_ = MetaData()
        self.trafos_ = {}
        self.col_trafos_ = {}
        # self.cleaners_ = {}
        self.col_types_ = {}
        self.train_splits_ = []
        self.test_splits_ = []

    def load_column_types(self, path: str) -> None:
        """ Reads column types from JSON.

            :param path (str)
                system path to JSON file with column information.

            :note
                This routine must be called before any transformation
                is attached to the cache.
        """
        column_type_info = {}
        with open(path, 'r', encoding='utf-8') as f:
            column_type_info = json.load(f)
        for col_name, col_type in column_type_info.items():
            self.col_types_[col_name] = col_type

    def attach_type_transformation(self,
                                   apply_to_type: str,
                                   trafo: ColumnTransform,
                                   # cleaner: ColumnCleaner,
                                   **trafo_args) -> None:
        """ Set transform for specified data type.

            @param trafo: column transformation type
                A column transformation (see transformations.py)
                This argument must be passed as a type, NOT as an instance.
            @param apply_to_type: str
                Column type to apply the transformation to.
                See `type_infer` for supported types.
            @param trafo_args: dict (defaults to empty dict)
                dictionary with transformation arguments.
        """
        for colname, coltype in self.col_types_.items():
            # ignore forced transformations
            if colname in self.col_trafos_:
                continue
            if coltype == apply_to_type:
                self.trafos_[colname] = trafo(**trafo_args)
               #  self.cleaners_[colname] = cleaner()

    def attach_column_transformation(self, apply_to_col: str,
                                     trafo: ColumnTransform,
                                     # cleaner: ColumnCleaner,
                                     **trafo_args) -> None:
        """ Set transform for specified column.

            @param apply_to_col: str
                name of column to apply transformation to.
            @param trafo: ColumnTransform
                type of column transformation to apply.
                Argument must be passed as a type, NOT as instance.
            @param trafo_args:
                positional arguments of specific transform.

            This function overrides `attach_type_transformation` for
            the specific column.
        """
        for colname in self.col_types_:
            if colname == apply_to_col:
                self.col_trafos_[colname] = trafo(**trafo_args)
                # self.cleaners_[colname] = cleaner()

    def get_column_transform(self, col_name: str) -> ColumnTransform:
        """ Returns transformation associated to a column.

            @param col_name: str
                name of the column to retrieve transformation from.

            @raises:
                ValueError if column `col_name` does not have a transform
                attached to it.
        """
        if col_name not in self.trafos_:
            raise ValueError(f"column {col_name} does not have a transform.")

        return self.trafos_[col_name]

    def transform(self) -> None:
        """ Applies transformations to specified columns.

            This function must be called *after* `load_types()`.

            The transformed columns are stored into internal DataFrame
            `x_data` and `y_data`. While the order is preserved, names
            might vary depending on how the transform renames the columns.
        """
        # transform column-wise
        trafo_dataframes = []
        for col_name in self.data_.columns:
            col_data = self.data_[col_name]
            if col_name in self.col_trafos_:
                # cleaning step is left for future implementations
                # col_data = self.cleaners_[col_name](col_data)
                # transform
                col_data = self.col_trafos_[col_name](col_data)
            elif col_name in self.trafos_:
                # cleaning step is left for future implementations
                # col_data = self.cleaners_[col_name](col_data)
                # transform
                col_data = self.trafos_[col_name](col_data)
            trafo_dataframes.append(col_data)

        trafo_data = pandas.concat(trafo_dataframes, axis=1)

        if self.target_ is not None:
            y_cols = [c for c in trafo_data.columns if c.startswith(self.target_)]
            self.y_data_ = trafo_data.filter(regex=f"^{self.target_}_.*")
            self.x_data_ = trafo_data.drop(y_cols, axis=1)
        else:
            self.x_data_ = trafo_data

    def populate_metadata(self) -> None:
        """ Writes column information to internal MetaData object.

            This routine must be called *after* `transform()` in order
            for the MetaData object to track transformed types correctly.
        """
        # input meta-data
        running_index = 0
        for col_name in self.data_.columns:
            if self.target_ is not None:
                if col_name == self.target_:
                    continue
            # get basic column information
            coltype = self.col_types_[col_name]
            # get column transformation (None is valid)
            coltrafo = self.col_trafos_.get(col_name, None)
            coltrafo = self.trafos_.get(col_name, None)
            self.x_meta_data_.add_column(col_name,
                                         running_index,
                                         coltype, coltrafo)
            running_index += 1
        # output (target) meta-data
        if self.target_ is not None:
            coltype = self.col_types_[self.target_]
            coltrafo = self.col_trafos_.get(self.target_, None)
            coltrafo = self.trafos_.get(self.target_, None)
            self.y_meta_data_.add_column(self.target_, 0, coltype, coltrafo)

    def build_splits(self,
                     splits: list = [{'global_fraction': 1.0,
                                      'val_fraction': 0.2}],
                     random_state: int = 42,
                     stratified: bool = False) -> None:
        """ Performs stratified splitting of the data.

            @param splits: list of dictionaries
                must have at least one entry with keywords

                'val_fraction':
                    floating point number between 0.0 and 1.0
                    to specify the fraction of the dataset used
                    for validation. Defaults to 0.2

                'global_fraction':
                    as 'val_fraction', but specifying the fraction
                    of the dataset used in global terms (for training
                    and validation) Defaults to 1.0

            This routine takes into account different class weights when
            performing the split by applying a stratified sampling scheme
            that preserves the probability distribution of the data for
            categorical and binary targets. For numerical targets, uniform
            sampling is used.
        """
        if self.target_ is None or (not stratified):
            X = self.x_data_
            for split_info in splits:
                val_fraction = split_info.get('val_fraction', 0.2)
                global_fraction = split_info.get('global_fraction', 1.0)
                idx_cutoff = int(len(X) * global_fraction)
                idx = numpy.arange(0, idx_cutoff)
                train_index, test_index = train_test_split(
                                            idx,
                                            test_size=val_fraction,
                                            shuffle=True,
                                            random_state=random_state)
                self.train_splits_.append(train_index)
                self.test_splits_.append(test_index)
        else: 
            X = self.x_data_
            Y = self.y_data_
            target = self.y_meta_data_.get_columns()[0]
            target_type = self.y_meta_data_.get_column_type(target)
            if target_type == 'categorical' and stratified:
                for split_info in splits:
                    val_fraction = split_info.get('val_fraction', 0.2)
                    global_fraction = split_info.get('global_fraction', 1.0)

                    idx_cutoff = int(len(X) * global_fraction)
                    Xs = X[:idx_cutoff]
                    Ys = Y[:idx_cutoff]

                    msss = \
                        MultilabelStratifiedShuffleSplit(n_splits=1,
                                                        test_size=val_fraction,
                                                        random_state=random_state)
                    for train_index, test_index in msss.split(Xs, Ys):
                        self.train_splits_.append(train_index)
                        self.test_splits_.append(test_index)

    def to_hdf5(self, path: str) -> None:
        """ Writes data to HDF5 file.

            @param path: str
                system path to file.

            Warning: this routine overwrites the file if it already exists.

            Resulting HDF5 file has two datasets called 'x' and 'y'.
            Dataset 'x' holds input data, and dataset 'y' holds target data.
            The metadata for each dataset is attached to `attrs`.
            as well.
        """
        if os.path.isfile(path):
            h5file = h5py.File(path, mode='r', libver='latest')
            h5file.close()
        h5file = h5py.File(path, mode='w')
        # write raw data
        x_data = self.x_data_.values
        h5file.create_dataset('x', x_data.shape, data=x_data, track_order=True)
        if self.target_ is not None:
            y_data = self.y_data_.values
            h5file.create_dataset('y', y_data.shape, data=y_data, track_order=True)
            h5file['y'].attrs.update(self.y_meta_data_)
        # write meta-data
        h5file['x'].attrs.update(self.x_meta_data_)
        # write splits
        i = 0
        for train_idx, test_idx in zip(self.train_splits_, self.test_splits_):
            h5file.create_dataset(f'train_split_{i}', train_idx.shape,
                                  data=train_idx)
            h5file.create_dataset(f'test_split_{i}', test_idx.shape,
                                  data=test_idx)
            i = i + 1
        h5file.attrs['n_splits'] = numpy.asarray([i, ])
        # close file
        h5file.close()

    @classmethod
    def from_hdf5(cls, path: str, has_target: bool = True):
        """ Reads cache from HDF5 file.

            Internal dataframe reference is lost when loading a cache from
            HDF5, as saving the data frame in this format is not supported
            given the possibility of having heterogenous data types.

            @param path: str
                system path to file.
        """
        h5file = h5py.File(path, mode='r', swmr=True)
        # instantiate dummy cache
        self = cls(pandas.DataFrame(), 'unknown', ignore=[])
        # read meta-data
        self.x_meta_data_.from_hdf5_attributes(h5file['x'].attrs)
        if has_target:
            self.y_meta_data_.from_hdf5_attributes(h5file['y'].attrs)
            # recover target name
            self.target_ = next(iter(self.y_meta_data_))
        # read splits
        n_splits = int(h5file.attrs['n_splits'][0])
        for i in range(n_splits):
            self.train_splits_.append(h5file[f'train_split_{i}'])
            self.test_splits_.append(h5file[f'test_split_{i}'])
        # read data
        self.data_ = None
        x_data = h5file['x']
        if has_target:
            y_data = h5file['y']
        # load transformations applied to input data
        for col_name in self.x_meta_data_:
            idx = self.x_meta_data_.get_column_index(col_name)
            trafo_args = self.x_meta_data_.get_argument_data(col_name)
            if len(trafo_args) > 0:
                trafo_params = self.x_meta_data_.get_parameter_data(col_name)
                trafo_name = ''.join(trafo_args['name'])
                # TODO: calling `eval` is dangerous. This must be changed
                #       by a safer method!
                trafo = eval(f'{trafo_name}()')
                trafo.from_metadata(trafo_args, trafo_params)
                self.trafos_[col_name] = trafo
            x_data[col_name] = h5file['x'][()][idx, :]
        if has_target:
            # load transformations applied to target data
            for col_name in self.y_meta_data_:
                idx = self.x_meta_data_.get_column_index(col_name)
                trafo_args = self.y_meta_data_.get_argument_data(col_name)
                if len(trafo_args) > 0:
                    trafo_params = self.y_meta_data_.get_parameter_data(col_name)
                    trafo_name = ''.join(trafo_args['name'])
                    # TODO: calling `eval` is dangerous. This must be changed
                    #       by a safer method!
                    trafo = eval(f'{trafo_name}()')
                    trafo.from_metadata(trafo_args, trafo_params)
                    self.trafos_[col_name] = trafo
            y_data[col_name] = h5file['x'][()][idx, :]

        self.x_data_ = x_data
        self.y_data_ = y_data

        return self
