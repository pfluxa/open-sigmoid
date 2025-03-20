import time
import json
from copy import copy

import numpy
import pandas
import torch
import torch.multiprocessing as mp
# mp.set_start_method('spawn')

from torchmetrics.regression import MeanSquaredError


# data files
from sigmoid.preprocessing.coordinate_files import Cache
from sigmoid.preprocessing.datasets import StandardDataset
from sigmoid.preprocessing.datasets import MixedTypesAutoencoderDataset
from sigmoid.preprocessing.dataloaders import StandardLoader
# column transforms
# from sigmoid.preprocessing.transformations import Passthrough
# from sigmoid.preprocessing.transformations import MinMaxTransform
from sigmoid.preprocessing.transformations import MinMaxTransform, Passthrough, QuantileTransform
from sigmoid.preprocessing.transformations import CategoricalAsOrdinal
from sigmoid.preprocessing.transformations import CategoricalAsOneHot
# models
from sigmoid.auto_encoding.models import AutoEmbedderWrapper
# switch
from sigmoid.switching.models import FCSwitch
# model scaler
from sigmoid.model_scaling.pools import StochasticPool

from sigmoid.analysis.clustering import ClusterFinder

class Skill(torch.nn.Module):

    def __init__(self, dim_embeddings, n_output):
        super(Skill, self).__init__()

        self.lin1_ = torch.nn.Linear(dim_embeddings, 50)
        self.act2_ = torch.nn.LeakyReLU()
        self.lin2_ = torch.nn.Linear(50, 50)
        self.act2_ = torch.nn.LeakyReLU()
        self.out_ = torch.nn.Linear(50, n_output)

    def forward(self, x):

        x = self.lin1_(x)
        x = self.lin2_(x)
        x = self.act2_(x)
        x = self.out_(x)

        return x

def read_data(path, nrows: int):
    """ Returns dataframe with a random sample of original data.
    """
    raw_dataframe = pandas.read_csv(path, low_memory=False, nrows=100000)
    dataframe = raw_dataframe.sample(n=nrows, replace=False)
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.dropna(axis=0, how='any')

    return dataframe


if __name__ == '__main__':

    E = 64
    B = 512
    N = 100000
    codec_dim = 8
    compute_device = torch.device('cuda')
    #torch.autograd.set_detect_anomaly(True)
    tic = time.time()

    data_path = '/home/pedro/projects/open-sigmoid/data/dielectron/dielectron.csv'

    # read, transform and write raw data into cache
    data_frame = read_data(data_path, nrows=N)

    # create local cache
    cache = Cache(data_frame, 'M', ignore=['Run', 'Event'])
    cache.load_column_types('/home/pedro/projects/open-sigmoid/data/dielectron/dielectron_types.json')
    cache.attach_type_transformation('numerical', QuantileTransform)
    cache.attach_type_transformation('categorical', CategoricalAsOrdinal)
    cache.transform()
    cache.populate_metadata()
    cache.build_splits()
    print(cache.x_data_.head())
    cache.to_hdf5('./temp.h5')

    # create dataset and data loaders
    print("setting up datasets... ")
    dataset = MixedTypesAutoencoderDataset('./temp.h5', cache_all=True)

    print("setting up dataloaders...")
    dataloader = StandardLoader(dataset)
    train_loader = dataloader.get_train_loader(split_id=0, num_workers=8, batch_size=B)
    test_loader = dataloader.get_test_loader(split_id=0, num_workers=8, batch_size=B)

    # create model
    print("setting up autoencoder... ")
    n_x_num = len(dataset.get_input_numerical_columns())
    c_x_cat = dataset.get_cardinalities()
    max_emb_size = max(c_x_cat)
    emb_sizes = []
    for c in c_x_cat:
        emb_sizes.append((c, E))
    parameters = {
        "hidden_layers": [[100, 1000], [1000, 100], [100, codec_dim]],
        "activation": "leaky_relu",
        "bias": True,
        "codec_dim": codec_dim,
        "width_multiplier": 0.25
    }
    ae = AutoEmbedderWrapper(parameters, n_x_num, emb_sizes)
    ae.set_device(compute_device)
    ae.set_kld_weight(B/N)
    ae.to(compute_device)

    print("training autoencoder...")
    ae.fit(train_loader, test_loader, n_epochs=100)

    # encode testing set
    test_encoded = ae.encode_data(test_loader)
    train_encoded = ae.encode_data(train_loader)
    test_encoded = test_encoded.cpu().numpy().reshape((-1, codec_dim))
    train_encoded = train_encoded.cpu().numpy().reshape((-1, codec_dim))

    # find clusters
    clusterer = ClusterFinder(
        min_samples=5,
        min_cluster_size=20,
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.0,
        n_jobs=24,
        metric='euclidean',
        store_centers='medoid'
    )
    test_labels = clusterer.find_clusters(test_encoded)
    # clusterer.elbow_kmeans(test_encoded)
    clusterer.plot_clusters(test_encoded, test_labels, prefix='testing_set')
    toc = time.time()
    print(f'elapsed time clustering using HDBSCAN: {toc - tic:.4f}')

    # filter out noisy samples
    labels = test_labels
    cluster_ids = numpy.unique(labels)
    n_clusters = len(cluster_ids)
    if -1 in cluster_ids:
        n_clusters -= 1
    print("number of clusters found:", n_clusters)
    if n_clusters == 0:
        raise RuntimeError("[CRITICAL] No clusters found.")
    noise_msk = labels == -1
    print(f"noisy samples = {numpy.sum(noise_msk)} total samples = {test_encoded.shape[0]}")
    labels = numpy.expand_dims(labels, axis=1)
    labels = labels[~noise_msk, :]
    test_encoded = test_encoded[~noise_msk, :]

    #create dummy data frame with synthetic labels
    col_names = []
    for i in range(test_encoded.shape[1]):
        col_names.append(f'codec_{i}')
    codec_column_names = copy(col_names)
    col_names.append('clusterId')
    # create json with data types
    column_types = {}
    for col_name in codec_column_names:
        column_types[col_name] = 'numerical'
    column_types['clusterId'] = 'categorical'
    with open('/home/pedro/projects/open-sigmoid/data/dielectron/encoded_data_types.json', 'w', encoding='utf-8') as f:
        json.dump(column_types, f, ensure_ascii=False, indent=4)

    data_enc = numpy.concatenate([test_encoded, labels], axis=1)
    switch_dataframe = pandas.DataFrame(
        data=data_enc,
        columns=list(column_types.keys()))
    print(switch_dataframe)
    # compute cluster weights
    switch_weights = switch_dataframe['clusterId'].value_counts(
        normalize=True, sort=False).values
    switch_weights = numpy.asarray(switch_weights)
    print("switch weights:", switch_weights)

    # setup cache to train switch
    switch_cache = Cache(switch_dataframe, 'clusterId', ignore=[])
    switch_cache.load_column_types('/home/pedro/projects/open-sigmoid/data/dielectron/encoded_data_types.json')
    switch_cache.attach_type_transformation('categorical', CategoricalAsOneHot)
    switch_cache.transform()
    switch_cache.populate_metadata()
    switch_cache.build_splits()
    print(switch_cache.x_data_.head())
    print(switch_cache.y_data_.head())
    switch_cache.to_hdf5('./temp_switch_cache.h5')

    # setup dataset and data loaders to train switch
    switch_dataset = StandardDataset('./temp_switch_cache.h5', cache_all=True)
    switch_loader = StandardLoader(switch_dataset)
    switch_train_loader = switch_loader.get_train_loader(split_id=0, batch_size=64)
    switch_test_loader = switch_loader.get_test_loader(split_id=0, batch_size=64)

    # build and train switch
    switch = FCSwitch(codec_dim, n_clusters)
    switch.set_device(compute_device)
    switch.build()
    switch.fit(switch_train_loader, switch_test_loader, 10)

    # print('CREATING SCALER CACHE')
    scaler_dataset = MixedTypesAutoencoderDataset('./temp.h5', cache_all=True)
    scaler_dataset.toggle_return_y()
    scaler_loader = StandardLoader(scaler_dataset)
    scaler_train_loader = scaler_loader.get_train_loader(split_id=0, shuffle=True, batch_size=256, num_workers=4)
    scaler_test_loader = scaler_loader.get_test_loader(split_id=0, shuffle=False, batch_size=256, num_workers=4)

    # build skill
    skill = Skill(ae.get_n_input(), 1)
    skill.to(compute_device)

    # specialist = StochasticPool(ae, 'regression')
    specialist = StochasticPool('regression')
    specialist.set_device(compute_device)
    specialist.set_cooldown_scale(10)
    specialist.set_autoencoder(ae)
    specialist.set_switch(switch)
    specialist.set_switch_balancing(switch_weights)
    # set specialist skill
    specialist.set_skills(skill, n_clusters, torch.nn.MSELoss())
    specialist.set_skill_metric(MeanSquaredError)
    # train specialist
    specialist.fit(scaler_train_loader, scaler_test_loader, n_epochs=100)

    # report output on test data
    gt, pr = specialist.evaluate(scaler_test_loader)

    pred_df = pandas.DataFrame()
    pred_df['ground_truth'] = gt.ravel()
    pred_df['prediction'] = pr.ravel()
    # pred_df['routing'] = rt

    pred_df.to_csv('predictions.csv')

    toc = time.time()
    print("# WALL TIME IN SECONDS: ", toc - tic)
