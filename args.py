import argparse
import warnings
import os
import torch
import sys
import time

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

# 2758 APK数 301 API数
APK_NUM = 2758
API_NUM = 301
ALL_NUM = APK_NUM + API_NUM
apk_set = set([i for i in range(APK_NUM)])
api_set = set([i for i in range(APK_NUM, ALL_NUM)])

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument("--log_file", type=str,
                    default="log/{}.txt".format(time.strftime('%Y%m%d%H%M%S', time.localtime())))
parser.add_argument('--input_dim', type=int, default=411, help='输入维度')
parser.add_argument('--hidden_dim', type=list, default=[128, 2], help='隐藏单元嵌入维度')
parser.add_argument('--num_neighbors_list', type=list, default=[5, 10],help='每层采样邻居的节点数')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--num_batch_per_epoch', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--pkl',default='411_dataset_w2v_api.pkl', type=str)

"""==【return args&dev】=="""
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# seed_torch(1206)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
