import os
from collections import Counter

import networkx as nx

import itertools
import math
from collections import defaultdict
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import utils
from utils import print_graph_detail

class BuildGraph:
    def __init__(self, dataset):
        self.graph_path = "api_data/graph"
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        self.word2id = dict()  # 单词映射
        self.id2word = dict()  # 算出敏感系数有反映射查找
        self.api_epio = {k.lower():v for k,v in utils.read_json("api_data/process/api_epio_dict.json").items()}
        self.dataset = dataset
        print(f"\n==> 现在的数据集是:{dataset}<==")

        self.g = nx.Graph()

        # self.content = f"{clean_corpus_path}/{dataset}.txt"
        self.content = r"E:\sln\graphsage\corpus\apk_api_corpus.txt"
        self.get_tfidf_edge()
        self.get_word2vec_edge()
        self.save()

    def get_word2vec_edge(self):
        api_pairs_cocurrence = utils.read_json("api_data/process/class_api_pairs_cocorrence.json")['api_pairs_cocurrence']
        w2v_type = 'cos_para'
        for api_pair in api_pairs_cocurrence:
            api_ind1 = self.node_num + self.word2id[api_pair['pair'][0].lower()]
            api_ind2 = self.node_num + self.word2id[api_pair['pair'][1].lower()]
            weight = api_pair[w2v_type]
            self.g.add_edge(api_ind1,api_ind2,weight=weight*8.76+5.42);


    def get_tfidf_edge(self):
        # 获得tfidf权重矩阵（sparse）和单词列表
        tfidf_vec = self.get_tfidf_vec()

        count_lst = list()  # 统计每个句子的长度
        for ind, row in tqdm(enumerate(tfidf_vec),
                             desc="generate tfidf edge"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                api_word = self.id2word[col_ind]
                epio = self.api_epio[api_word][0]
                word_ind = self.node_num + col_ind
                self.g.add_edge(ind, word_ind, weight=1)
                count += 1
            count_lst.append(count)

        print_graph_detail(self.g)

    def get_tfidf_vec(self):
        """
        学习获得tfidf矩阵，及其对应的单词序列
        :param content_lst:
        :return:
        """
        start = time()
        text_tfidf = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])

        tfidf_vec = text_tfidf.fit_transform(open(self.content, "r"))

        self.tfidf_time = time() - start
        print("tfidf time:", self.tfidf_time)
        print("tfidf_vec shape:", tfidf_vec.shape)
        print("tfidf_vec type:", type(tfidf_vec))

        self.node_num = tfidf_vec.shape[0]

        # 映射单词
        vocab_lst = text_tfidf["vect"].get_feature_names()
        print("vocab_lst len:", len(vocab_lst))
        for ind, word in enumerate(vocab_lst):
            self.word2id[word] = ind
            self.id2word[ind] = word

        self.vocab_lst = vocab_lst

        return tfidf_vec

    def save(self):
        print("total time:", self.tfidf_time)
        # nx.write_weighted_edgelist(self.g,
        #                            f"{self.graph_path}/{self.dataset}.txt")
        nx.write_weighted_edgelist(self.g,
                                   f"{self.graph_path}/w2v/{self.dataset}_class_cos_api.txt")
        # nx.write_weighted_edgelist(self.g,
        #                            f"{self.graph_path}/base_epio.txt")
        print("\n")


def main():
    # BuildGraph("mr")
    # BuildGraph("ohsumed")
    # BuildGraph("R52")
    BuildGraph("dataset")
    # BuildGraph("20ng")
    # BuildGraph("dataset")


if __name__ == '__main__':
    main()