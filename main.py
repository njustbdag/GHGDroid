# -*- coding: utf-8 -*-


import utils
import torch
from process import load_data
import numpy as np
import torch.nn as nn
import torch.optim as optim
from graphsage import GraphSage
from sampling import multihop_sampling
import time

from collections import namedtuple
INPUT_DIM = 411    # 输入维度
# INPUT_DIM = 13337    # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 2]   # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [5, 10]   # 每阶采样邻居的节点数
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
BTACH_SIZE = 1    # 批处理大小
EPOCHS = 1000
NUM_BATCH_PER_EPOCH = 20    # 每个epoch循环的批次数
LEARNING_RATE = 0.01    # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])

# pkl = "dialog_dataset.pkl"
pkl = '411_dataset_w2v_api.pkl'
# pkl = '411_dataset_class_cos_api_mul_e1.pkl'
data = load_data(False,pkl)
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
x = data.x
print("x = ", x.shape)
print("x type = ", type(x))
print(x[[0,1]])

train_index = np.where(data.train_mask)[0]
train_label = data.y
test_index = np.where(data.test_mask)[0]
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                  num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
print("model = ",model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

logger = utils.Logging('{}_{}.log'.format(pkl.split('.')[0],time.strftime("%Y%m%d%H%M%S", time.localtime())))
def train():
    model.train()
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
            # print("batch_src_index = " , batch_src_index)
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)

            # print("batch_sampling_result = ", batch_sampling_result)

            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
            # print("batch_sampling_x = ",batch_sampling_x)
            # print("batch_sampling_x size = ",len(batch_sampling_x))
            # print("batch_sampling_x 1 size = ",batch_sampling_x[0].shape)
            # print("batch_sampling_x 2 size = ",batch_sampling_x[1].shape)
            # print("batch_sampling_x 3 size = ",batch_sampling_x[2].shape)


            batch_train_logits = model(batch_sampling_x)
            loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            print("Epoch {:03d} Batch {:03d} Loss: {:.8f}".format(e, batch, loss.item()))
        # start = time.time()
        test(e)
        # end = time.time()
        # print(end-start)
        # model_name = "test.pt"
        # if e == 10:
        #     torch.save(model, model_name)
        #     break;


def test(e):
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        # print(test_x[0].shape,test_x[1].shape,test_x[2].shape)
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        # print(test_label)
        predict_y = test_logits.max(1)[1]
        # print(predict_y.shape,predict_y)
        # print(test_label.shape,test_label.shape)
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        f1,precision,recall = utils.macro_f1(predict_y,test_label)
        # print("Test Accuracy: ", accuarcy)

        desc = {
            "epoch":e,
            "acc": accuarcy,
            "macro_f1": f1,
            "precision": precision,
            "recall": recall,
        }
        utils.set_description(logger,desc)
def val():
    model = torch.load('test.pt')
    test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
    test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
    test_logits = model(test_x)
    test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
    predict_y = test_logits.max(1)[1]
    accuarcy = torch.eq(predict_y, test_label).float().mean().item()
    f1, precision, recall = utils.macro_f1(predict_y, test_label)
    desc = {
        "acc": accuarcy,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
    }
    utils.set_description(logger, desc)
    return f1

if __name__ == '__main__':
    logger.info("----------运行的数据为{}---------".format(pkl))
    train()
    # start = time.time()
    # val()
    # end = time.time()
    # print(end - start)


