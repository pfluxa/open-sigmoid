import time
import json
from copy import copy

import math
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


def read_data(path, nrows: int):
    """ Returns dataframe with a random sample of original data.
    """
    raw_dataframe = pandas.read_csv(path, low_memory=False, nrows=100000)
    dataframe = raw_dataframe.sample(n=nrows, replace=False)
    dataframe = dataframe.reset_index(drop=True)

    return dataframe

def cleanse(dataframe):
    """ Specific for airline delays dataset.

    0   FL_DATE              100000 non-null  object
    1   OP_CARRIER           100000 non-null  object
    2   OP_CARRIER_FL_NUM    100000 non-null  int64
    3   ORIGIN               100000 non-null  object
    4   DEST                 100000 non-null  object
    5   CRS_DEP_TIME         100000 non-null  int64
    6   DEP_TIME             98653 non-null   float64
    7   DEP_DELAY            98653 non-null   float64
    8   TAXI_OUT             98595 non-null   float64
    9   WHEELS_OFF           98595 non-null   float64
    10  WHEELS_ON            98450 non-null   float64
    11  TAXI_IN              98450 non-null   float64
    12  CRS_ARR_TIME         100000 non-null  int64
    13  ARR_TIME             98450 non-null   float64
    14  ARR_DELAY            98253 non-null   float64
    15  CANCELLED            100000 non-null  float64
    16  CANCELLATION_CODE    1450 non-null    object
    17  DIVERTED             100000 non-null  float64
    18  CRS_ELAPSED_TIME     100000 non-null  float64
    19  ACTUAL_ELAPSED_TIME  98253 non-null   float64
    20  AIR_TIME             98253 non-null   float64
    21  DISTANCE             100000 non-null  float64
    22  CARRIER_DELAY        26954 non-null   float64
    23  WEATHER_DELAY        26954 non-null   float64
    24  NAS_DELAY            26954 non-null   float64
    25  SECURITY_DELAY       26954 non-null   float64
    26  LATE_AIRCRAFT_DELAY  26954 non-null   float64
    27  Unnamed: 27          0 non-null       float64

    """
    # # date to day of the year
    # dataframe['FL_DATE'] = pandas.to_datetime(dataframe['FL_DATE'])
    # dataframe['day_of_year'] = dataframe['FL_DATE'].dt.dayofyear
    # dataframe.drop(['FL_DATE'], axis=1, inplace=True)
    # nan-filling on non critical columns
    dataframe['CANCELLATION_CODE'] = dataframe['CANCELLATION_CODE'].fillna('Z')
    dataframe['CARRIER_DELAY'] = dataframe['CARRIER_DELAY'].fillna(-1.0)
    dataframe['WEATHER_DELAY'] = dataframe['WEATHER_DELAY'].fillna(-1.0)
    dataframe['NAS_DELAY'] = dataframe['NAS_DELAY'].fillna(-1.0)
    dataframe['SECURITY_DELAY'] = dataframe['SECURITY_DELAY'].fillna(-1.0)
    dataframe['LATE_AIRCRAFT_DELAY'] = dataframe['LATE_AIRCRAFT_DELAY'].fillna(-1.0)
    # drop dummy column
    dataframe.drop(['Unnamed: 27'], axis=1, inplace=True)

    # filter rows with NaN on critical columns
    dataframe.dropna(axis=0, how='any', inplace=True,
        subset=[
            'DEP_TIME',
            'DEP_DELAY',
            'TAXI_OUT',
            'WHEELS_OFF',
            'WHEELS_ON',
            'TAXI_IN',
            'ARR_TIME',
            'ARR_DELAY',
            'ACTUAL_ELAPSED_TIME',
            'AIR_TIME'])

    dataframe = dataframe.dropna(axis=0, how='any', inplace=False)
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe

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

if __name__ == '__main__':

    B = 128
    N = 120000
    codec_dim = 10
    compute_device = torch.device('cuda')
    #torch.autograd.set_detect_anomaly(True)
    tic = time.time()

    data_path = '/home/pedro/projects/open-sigmoid/data/airline_delays/data.csv'
    # read, transform and write raw data into cache
    data_frame = read_data(data_path, nrows=N)
    data_frame = cleanse(data_frame)

    # create local cache
    splits = [
        {
            'training': 0.6,
            'testing': 0.2,
            'validation': 0.1,
            'clustering': 0.1,
            'ignore': 0.5,
        },
        {
            'training': 0.6,
            'testing': 0.2,
            'validation': 0.1,
            'clustering': 0.1,
        }
    ]
    cache = Cache(data_frame, 'ARR_DELAY', ignore=['FL_DATE', 'CANCELLATION_CODE'])
    cache.load_column_types('/home/pedro/projects/open-sigmoid/data/airline_delays/data_types.json')
    cache.attach_type_transformation('numerical', MinMaxTransform)
    cache.attach_type_transformation('categorical', CategoricalAsOrdinal)
    cache.transform()
    cache.populate_metadata()
    cache.build_splits(splits)
    cache.to_hdf5('./temp.h5')

    # create dataset and data loaders
    print("setting up datasets... ")
    dataset = MixedTypesAutoencoderDataset('./temp.h5', cache_all=True)

    print("setting up dataloaders...")
    dataloader = StandardLoader(dataset)
    train_loader = dataloader.get_loader("training", 0, batch_size=B)
    test_loader = dataloader.get_loader("testing", 0, batch_size=B, shuffle=False)
    val_loader = dataloader.get_loader("validation", 0, batch_size=B, shuffle=False)
    clus_loader = dataloader.get_loader("clustering", 0, batch_size=B)
    
    # create model
    print("setting up autoencoder... ")
    n_x_num = len(dataset.get_input_numerical_columns())
    c_x_cat = dataset.get_cardinalities()
    max_emb_size = max(c_x_cat)
    print("max cardinality:", max_emb_size)
    emb_sizes = [(n_x_num, int(math.sqrt(n_x_num) + 1))]
    for c in c_x_cat:
        emb_sizes.append((c, int(math.sqrt(c) + 1)))
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
    # ae.set_kld_weight(0.0)
    ae.to(compute_device)

    print("training autoencoder...")
    ae.fit(train_loader, test_loader, n_epochs=200)
    val_score = ae.evaluate(val_loader)
    print(f"MSE on validation set: {val_score:4.4e}")
    
    # encode clustering set
    test_encoded = ae.encode_data(clus_loader)
    # maximum number of points is 40000 because my computer is too crappy
    test_encoded = test_encoded.cpu().numpy().reshape((-1, codec_dim))[0:40000]

    # find clusters
    clusterer = ClusterFinder(
        min_samples=5,
        min_cluster_size=5,
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.0,
        # n_jobs=12,
        metric='precomputed'
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
    with open('/home/pedro/projects/open-sigmoid/data/airline_delays/encoded_data_types.json', 'w', encoding='utf-8') as f:
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
    switch_cache.load_column_types('/home/pedro/projects/open-sigmoid/data/airline_delays/encoded_data_types.json')
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
    switch_train_loader = switch_loader.get_loader("training", split_id=0, batch_size=B, shuffle=True)
    switch_test_loader = switch_loader.get_loader("testing", split_id=0, batch_size=B)

    # build and train switch
    switch = FCSwitch(codec_dim, n_clusters)
    switch.set_device(compute_device)
    switch.build()
    switch.fit(switch_train_loader, switch_test_loader, 50)

    # print('CREATING SCALER CACHE')
    scaler_dataset = MixedTypesAutoencoderDataset('./temp.h5', cache_all=True)
    scaler_dataset.toggle_return_y()
    scaler_loader = StandardLoader(scaler_dataset)
    scaler_train_loader = scaler_loader.get_loader("training", split_id=1, shuffle=True, batch_size=B, num_workers=4)
    scaler_test_loader = scaler_loader.get_loader("testing", split_id=1, shuffle=False, batch_size=B, num_workers=4)
    scaler_val_loader = scaler_loader.get_loader("validation", split_id=1, shuffle=False, batch_size=1, num_workers=4)

    # build skill
    skill = Skill(ae.get_n_input(), 1)
    skill.to(compute_device)

    # specialist = StochasticPool(ae, 'regression')
    specialist = StochasticPool('regression')
    specialist.set_device(compute_device)
    specialist.set_cooldown_scale(20)
    specialist.set_autoencoder(ae)
    specialist.set_switch(switch)
    specialist.set_switch_balancing(switch_weights)
    # set specialist skill
    specialist.set_skills(skill, n_clusters, torch.nn.MSELoss())
    specialist.set_skill_metric(MeanSquaredError)
    # train specialist
    specialist.fit(scaler_train_loader, scaler_test_loader, n_epochs=100)

    # report output on test data
    gt, pr = specialist.evaluate(scaler_val_loader)

    pred_df = pandas.DataFrame()
    pred_df['ground_truth'] = gt.ravel()
    pred_df['prediction'] = pr.ravel()
    # pred_df['routing'] = rt

    pred_df.to_csv('predictions.csv')

    toc = time.time()
    print("# WALL TIME IN SECONDS: ", toc - tic)
