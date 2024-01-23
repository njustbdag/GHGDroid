# -*- coding: utf-8 -*-


import utils
import torch
from process import load_data
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import GraphSage
from sampling import multihop_sampling
import time
from collections import namedtuple
from args import args,DEVICE

from args import args

print(args)
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])
data = load_data(False,args.pkl)
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
x = data.x
print("x = ", x.shape)

train_index = np.where(data.train_mask)[0]
train_label = data.y
test_index = np.where(data.test_mask)[0]
model = GraphSage(input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                  num_neighbors_list=args.num_neighbors_list).to(DEVICE)
print("model = ",model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

logger = utils.Logging(args.log_file)


def train():
    model.train()
    for e in range(args.epochs):
        for batch in range(args.num_batch_per_epoch):
            batch_src_index = np.random.choice(train_index, size=(args.batch_size,))
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
            # print("batch_src_index = " , batch_src_index)
            batch_sampling_result = multihop_sampling(batch_src_index, args.num_neighbors_list, data.adjacency_dict)

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
        test_sampling_result = multihop_sampling(test_index, args.num_neighbors_list, data.adjacency_dict)
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
    test_sampling_result = multihop_sampling(test_index, args.num_neighbors_list, data.adjacency_dict)
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
    print("start")

    logger.info("----------运行的数据为{}---------".format(args.pkl))
    train()
    start = time.time()
    val()
    end = time.time()
    print(end - start)


