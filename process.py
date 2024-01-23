import os
import pickle
import joblib
import os.path as osp
import utils
from collections import defaultdict
import networkx as nx
import numpy as np
from collections import namedtuple

N = 13337  # 图中有apk12926+api411 共13337个节点

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])


def get_graph():
    with open("api_data/graph/w2v/dataset_class_cos_api.txt", 'r', encoding='utf8') as f:
        data = f.read().split('\n')
    edge_map = defaultdict(list)
    for edge in data:
        edge = edge.split(' ')
        # print(edge)
        edge_map[int(edge[0])].append(int(edge[1]))
        edge_map[int(edge[1])].append(int(edge[0]))
    utils.write_json("data/graph/w2v_graph.json", edge_map)


def load_data(rebuild, save_file):
    data_root = "data/pkl/"
    save_file = osp.join(data_root, save_file)
    if osp.exists(save_file) and not rebuild:
        print("Using Cached file: {}".format(save_file))
        data = joblib.load(open(save_file, "rb"))
    else:
        data = preprocess_data()
        with open(save_file.format(save_file), "wb") as f:
            joblib.dump(data, f)
        print("Cached file: {}".format(save_file))
    return data


def preprocess_data():
    # 纯对角矩阵 无特征 [13337,13337]

    nfeat_dim = N
    # features
    # dia = [1 for i in range(nfeat_dim)]
    # features = np.diag(dia)

    # 异构图特征 [13337,13337]

    # graph = nx.read_weighted_edgelist("api_data/graph/dataset_mul_e1.txt"
    #                                   , nodetype=int)
    # utils.print_graph_detail(graph)
    # adj = nx.to_scipy_sparse_matrix(graph,
    #                                 nodelist=list(range(graph.number_of_nodes())),
    #                                 weight='weight',
    #                                 dtype=np.float)
    #
    # features = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = features.A

    # APK-API节点特征 [13337,411] 维
    # features = get_features()

    # APK-API API-API节点特征 [13337,411]维度
    features = get_features_plus()
    # print(features.shape)

    dataset = utils.read_file("api_data/dataset.txt")
    x = []
    y = []
    y_map = {0: [], 1: []}
    for idx, data in enumerate(dataset):
        data = data.split('\t')
        x.append(int(idx))
        y.append(int(data[1]))
        y_map[int(data[1])].append(idx)

    num_nodes = 12926
    train_mask = np.zeros(num_nodes, dtype=np.bool)
    val_mask = np.zeros(num_nodes, dtype=np.bool)
    test_mask = np.zeros(num_nodes, dtype=np.bool)

    train_ratio = 0.8
    test_ratio = 0.2
    val_ratio = 0
    train_index = list()
    test_index = list()
    # val_index = list()
    for ele in y_map:
        train_index.extend(y_map[ele][0:int(train_ratio * len(y_map[ele]))])
        # val_index.extend(
        #     y_map[ele][int(train_ratio * len(y_map[ele])):int((train_ratio + val_ratio) * len(y_map[ele]))])
        test_index.extend(y_map[ele][int((train_ratio + val_ratio) * len(y_map[ele])):])

    train_mask[np.array(train_index)] = True
    test_mask[np.array(test_index)] = True

    # val_mask[np.array(val_index)] = True
    val_mask = np.array([])


    x = np.array(x)
    y = np.array(y)

    adjacency_dict = utils.read_json("data/graph/w2v_graph.json")
    print("Node's feature shape: ", features.shape)
    print("Node's label shape: ", y.shape)
    print("Adjacency's shape: ", len(adjacency_dict))
    print("Number of training nodes: ", train_mask.sum())
    print("Number of validation nodes: ", val_mask.sum())
    print("Number of test nodes: ", test_mask.sum())
    return Data(x=features, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                adjacency_dict=adjacency_dict)


def get_features():
    graph_data = utils.read_file("api_data/graph/dataset.txt")

    result = np.zeros((13337, 411))
    for line in graph_data:
        line = line.split(" ")
        if int(line[0]) > 12925:
            result[int(line[1]), int(line[0]) - 12926] = line[2]
        else:
            result[int(line[0]), int(line[1]) - 12926] = line[2]
    for i in range(12926, 13337):
        result[i, i - 12926] = 1
    return result


def get_features_plus():
    graph_data = utils.read_file("api_data/graph/w2v/dataset_class_cos_api.txt")
    result = np.zeros((13337, 411))
    for line in graph_data:
        line = line.split(" ")
        if int(line[0]) > 12925 and int(line[1]) <= 12925:
            result[int(line[1]), int(line[0]) - 12926] = line[2]
        elif int(line[1]) > 12925 and int(line[0]) <= 12925:
            result[int(line[0]), int(line[1]) - 12926] = line[2]
        elif int(line[0]) > 12925 and int(line[1]) > 12925:
            result[int(line[0])-12926, int(line[1]) - 12926] = line[2]
        else:
            print(line)
    return result


if __name__ == '__main__':
    # get_data = load_data(False,"dataset_mul_e1.pkl")
    get_graph()
    # get_features()
