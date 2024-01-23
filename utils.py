import json
import numpy as np
import torch as th
import scipy.sparse as sp
import logging

def write_file(out,result):
    with open(out,"w",encoding="utf8") as f:
        f.write("\n".join(result))

def read_file(path):
    with open(path, "r", encoding="utf8") as f:
        data = f.read()
    return data.split("\n")

def write_json(out,dic):
    json_data = json.dumps(dic)
    with open(out,"w",encoding="utf8") as f:
        f.write(json_data)

def read_json(path):
    with open(path,"r",encoding="utf8") as f:
        data = json.load(f)
    return data

def print_graph_detail(graph):
    """
    格式化显示Graph参数
    :param graph:
    :return:
    """
    import networkx as nx
    dst = {"nodes"    : nx.number_of_nodes(graph),
           "edges"    : nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates" : nx.number_of_isolates(graph),
           "覆盖度"      : 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    print_table(dst)


def print_table(dst):
    table_title = list(dst.keys())
    from prettytable import PrettyTable
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    print(table)


def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return th.from_numpy(adj_normalized.A).float()


def set_description(logger,desc):
    string = ""
    for key, value in desc.items():
        if isinstance(value, int):
            string += f"{key}:{value} "
        else:
            string += f"{key}:{value:.4f} "
    logger.info(string)
    print(string)

def Logging(log_file):
    # 创建logger对象
    logger = logging.getLogger('result_logger')

    # 设置日志等级
    logger.setLevel(logging.DEBUG)

    # 追加写入文件a ，设置utf-8编码防止中文写入乱码
    test_log = logging.FileHandler(log_file, 'a', encoding='utf-8')

    # 向文件输出的日志级别
    test_log.setLevel(logging.DEBUG)

    # 向文件输出的日志信息格式
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')

    test_log.setFormatter(formatter)

    # 加载文件到logger对象中
    logger.addHandler(test_log)
    return logger


def macro_f1(pred, targ):
    tp = ((pred == 1) & (targ == 1)).sum().item()  # 预测为i，且标签的确为i的
    fp = ((pred == 1) & (targ != 1)).sum().item()  # 预测为i，但标签不是为i的
    tn = ((pred == 0) & (targ == 0)).sum().item()
    fn = ((pred == 0) & (targ != 0)).sum().item()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1, precision, recall

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def test():
    import joblib
    file = "data/pkl/411_dataset_mul_e1.pkl"
    data = joblib.load(open(file,"rb"))
    test_index = np.where(data.test_mask)[0]
    return test_index

if __name__ == '__main__':
    test_index = test().tolist()
    print(test_index)
    with open("test_index.txt",'w',encoding="utf8") as f:
        f.write(test_index)
    # write_file("test_index.txt",test_index.tolist())