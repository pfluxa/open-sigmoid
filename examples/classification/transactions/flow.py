from os import getcwd
import time
import time
import json
from copy import copy

import math
import numpy
import pandas
import torch
import torch.multiprocessing as mp
# mp.set_start_method('spawn')

from torchmetrics.classification import MulticlassConfusionMatrix


# data files
from sigmoid.preprocessing.coordinate_files import Cache
from sigmoid.preprocessing.datasets import StandardDataset
from sigmoid.preprocessing.datasets import MixedTypesAutoencoderDataset
from sigmoid.preprocessing.dataloaders import StandardLoader
# column transforms
# from sigmoid.preprocessing.transformations import Passthrough
# from sigmoid.preprocessing.transformations import MinMaxTransform
from sigmoid.preprocessing.transformations import MinMaxTransform
from sigmoid.preprocessing.transformations import Passthrough
from sigmoid.preprocessing.transformations import QuantileTransform
from sigmoid.preprocessing.transformations import CategoricalAsOrdinal
from sigmoid.preprocessing.transformations import CategoricalAsOneHot
# models
from sigmoid.nn.embeddings import PLE
from sigmoid.nn.embedders import Autoembedder
from sigmoid.auto_encoding.models import AutoEmbedderWrapper
# switch
from sigmoid.switching.models import FCSwitch
# model scaler
from sigmoid.model_scaling.pools import StochasticPool

from sigmoid.analysis.clustering import ClusterFinder

class Skill(torch.nn.Module):

    def __init__(self, dim_embeddings, n_output):
        super(Skill, self).__init__()

        print("n input = ", dim_embeddings)
        self.lin1_ = torch.nn.Linear(dim_embeddings, 32)
        self.act1_ = torch.nn.LeakyReLU()
        self.lin2_ = torch.nn.Linear(32, 32)
        self.act2_ = torch.nn.LeakyReLU()
        self.out_ = torch.nn.Linear(32, n_output)
        self.act_out_ = torch.nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.lin1_(x)
        x = self.act1_(x)
        x = self.lin2_(x)
        x = self.act2_(x)
        x = self.out_(x)
        x = self.act_out_(x)

        return x

def read_data(path, nrows: int):
    """ Returns dataframe with a random sample of original data.
    """
    df = pandas.read_csv(path, low_memory=False)
    # drop extra index
    df.drop('Unnamed: 0', axis=1, inplace=True)
    # filter frauds
    dff = df[df['isFraud'] > 0].copy()
    if len(dff) > nrows//2:
        dff = dff.sample(n=nrows//2)
    dfl = df[df['isFraud'] < 1].copy()
    dfl = dfl.sample(n=(nrows - len(dff)))
    dff = dff.reset_index(drop=True)
    dfl = dfl.reset_index(drop=True)
    # dataframe = pandas.concat([dff, dfl], axis=0).reset_index(drop=True)
    dataframe = dfl
    # shuffle
    dataframe = dataframe.sample(frac=1.0).reset_index(drop=True)
    # drop NaN
    dataframe = dataframe.dropna(axis=0, how='any')

    print(dataframe.nunique())
    print("n rows = ", len(df))
    print("n fraud = ", len(dff), pandas.unique(dff['isFraud']))
    print("n valid = ", len(dfl), pandas.unique(dfl['isFraud']))
    n_false = len(dff)
    n_true = len(dfl)

    return dataframe, n_true, n_false


if __name__ == '__main__':

    # E = 8
    B = 4096
    N = 500000
    codec_dim = 5
    compute_device = torch.device('cuda')
    torch.autograd.set_detect_anomaly(True)
    tic = time.time()

    data_path = '/home/pedro/projects/open-sigmoid/data/transactions/data.csv'
    # read, transform and write raw data into cache
    data_frame, n_true, n_false = read_data(data_path, nrows=N)
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
    # create local cache
    cache = Cache(data_frame,
                    'isFraud',
                    ignore=[
                      'customerId',
                      'merchantName',
                      'accountNumber',
                      'cardLast4Digits',
                      'currentExpDate_day',
                      'transactionDateTime_year'
                    ]
    )
    cache.load_column_types('/home/pedro/projects/open-sigmoid/data/transactions/column_types.json')
    cache.attach_column_transformation('isFraud', CategoricalAsOrdinal)
    cache.attach_type_transformation('numerical', Passthrough)
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
    train_loader = dataloader.get_loader("training", 0, batch_size=B, shuffle=True, num_workers=4)
    test_loader = dataloader.get_loader("testing", 0, batch_size=B, shuffle=False)
    val_loader = dataloader.get_loader("validation", 0, batch_size=B, shuffle=False)
    clus_loader = dataloader.get_loader("clustering", 0, batch_size=B, shuffle=False)

    # create model
    print("setting up autoencoder... ")

    idx_x_num = dataset.get_input_numerical_columns()
    num_ranges = []
    for idx in idx_x_num:
        l, h = dataset.get_column_min_max(idx)
        num_ranges.append((l, h))
    depths = []
    for idx, (l, h) in zip(idx_x_num, num_ranges):
        dx = (h - l) / dataset.get_column_nunique_values(idx)
        depths.append(int(-math.log2(dx)) + 1)

    cardinalities = dataset.get_cardinalities()

    parameters = {
        "decoder_dims": [
            {'in_dim': codec_dim, 'out_dim': 100},
            {'in_dim': 100, 'out_dim': 1000},
            {'in_dim': 1000, 'out_dim': 1000},
            {'in_dim': 1000, 'out_dim': 100},
        ],
        "codec_dim": codec_dim,
    }
    emb_model = Autoembedder(parameters)
    emb_model.build_embedding_layers(
        num_ranges, depths,
        cardinalities
    )
    emb_model.build_num_encoder()
    emb_model.build_cat_encoder()
    emb_model.build_decoder()

    ae = AutoEmbedderWrapper(parameters)
    ae.set_embedding_model(emb_model)
    ae.set_device(compute_device)
    # ae.set_kld_weight(B/N)
    # ae.set_kld_weight(0.0)
    ae.to(compute_device)

    print("training autoencoder...")
    ae.fit(train_loader, test_loader, n_epochs=50)
    val_score = ae.evaluate(val_loader)
    print(f"MSE on validation set: {val_score:4.4e}")

    # encode clustering set
    test_encoded = ae.encode_data(clus_loader)
    # maximum number of points is 40000 because my computer is too crappy
    test_encoded = test_encoded.cpu().numpy().reshape((-1, codec_dim))[0:40000]

    # a = 1.0
    # n = test_encoded.shape[0]
    # d = test_encoded.shape[1]
    # with open("./codecs.txt", "w") as f:
    #     f.write("FACTORIZED COMPLETE MULTICUT\n")
    #     f.write(f"{a}\n")
    #     f.write(f"{n} {d}\n")
    #     for enc in test_encoded:
    #         for x in enc[0:-1]:
    #             f.write(f"{x} ")
    #         x = enc[-1]
    #         f.write(f"{x}\n")

    # import sys
    # sys.exit(-1)

    # find clusters
    clusterer = ClusterFinder(
        min_samples=5,
        min_cluster_size=20,
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.01,
        n_jobs=12,
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
        print("[CRITICAL] No clusters found.")
        n_clusters += 1
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
    with open('/home/pedro/projects/open-sigmoid/data/transactions/encoded_data_types.json', 'w', encoding='utf-8') as f:
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
    switch_cache.load_column_types('/home/pedro/projects/open-sigmoid/data/transactions/encoded_data_types.json')
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
    scaler_train_loader = scaler_loader.get_loader("training", split_id=1, shuffle=True, batch_size=B)
    scaler_test_loader = scaler_loader.get_loader("testing", split_id=1, shuffle=False, batch_size=B)
    scaler_val_loader = scaler_loader.get_loader("validation", split_id=1, shuffle=False, batch_size=1)

    # build skill
    skill = Skill(ae.get_n_input(), 2)
    # skill = Skill(329, 2)
    skill.to(compute_device)

    specialist = StochasticPool('classification')
    specialist.set_device(compute_device)
    specialist.set_cooldown_scale(20)
    specialist.set_autoencoder(ae)
    specialist.set_switch(switch)
    specialist.set_switch_balancing(switch_weights)
    # set specialist skill
    specialist.set_skills(skill, n_clusters, torch.nn.CrossEntropyLoss())
    specialist.set_skill_metric(MulticlassConfusionMatrix, num_classes=2)
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
