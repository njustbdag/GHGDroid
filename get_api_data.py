import math
import os
from collections import defaultdict
# import gensim
# from gensim.models.word2vec import LineSentence
# from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import numpy as np
import utils
from tqdm import tqdm
import fnmatch
seed = 100

"""
给不同路径下的apk添加标签 
1：malware 0：beign
return: befistdir(benign_dir)
    print(malware_listore_dataset.txt
"""
def get_label_data():
    malware_dir = r"G:\sln\malware"
    benign_dir = r"G:\sln\benign"
    malware_list = os.listdir(malware_dir)
    benign_list = os.listdir(benign_dir)
    print(benign_list)
    result = []
    for malware in malware_list:
        result.append("{}\t{}".format(malware,1))
    for benign in benign_list:
        result.append("{}\t{}".format(benign,0))
    utils.write_file("api_data/process/before_dataset.txt", result)

'''
根据seed划分训练集和测试集 7:3
return: dataset.txt
'''
def split():
    with open("api_data/process/before_dataset.txt", 'r', encoding="utf8") as f:
        data = f.readlines()

    x = np.array([i.split("\t")[0].strip() for i in data])
    y = np.array([i.split("\t")[1].strip() for i in data])
    print(x.shape)
    print(y.shape)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,shuffle=True,random_state=seed)

    print(len(x_train),len(y_train),len(x_test),len(y_test))

    result_m = []
    for i in range(len(x_train)):
        if y_train[i] == '1':
            result_m.append("{}\t{}\t{}".format(x_train[i],'train',y_train[i]))
    with open("api_data/dataset2_91.txt", "w", encoding="utf8") as f:
        f.write("\n".join(result_m))
        print(len(result_m))
    result_b = []
    for i in range(len(x_train)):
        if y_train[i] == '0':
            result_b.append("{}\t{}\t{}".format(x_train[i],'train',y_train[i]))
    with open("api_data/dataset2_91.txt", "a", encoding="utf8") as f:
        f.write("\n")
        f.write("\n".join(result_b))
        print(len(result_b))

    result_m = []
    for i in range(len(x_test)):
        if y_test[i] == '1':
            result_m.append("{}\t{}\t{}".format(x_test[i],'test',y_test[i]))
    with open("api_data/dataset2_91.txt", "a", encoding="utf8") as f:
        f.write("\n")
        f.write("\n".join(result_m))
        print(len(result_m))
    result_b = []
    for i in range(len(x_test)):
        if y_test[i] == '0':
            result_b.append("{}\t{}\t{}".format(x_test[i],'test',y_test[i]))
    with open("api_data/dataset2_91.txt", "a", encoding="utf8") as f:
        f.write("\n")
        f.write("\n".join(result_b))
        print(len(result_b))

'''
为dataset.txt中的每个apk找到对应的apk$api_num.txt
code ( 1:malware 0:benign )
return: map {apk_name:apk_api_path}
'''
def get_path(dir_path,code):
    with open("api_data/process/before_dataset.txt", 'r', encoding="utf8") as f:
        data = f.readlines();

    api_txt_list = os.listdir(dir_path)
    corpus_file = [txt for txt in api_txt_list]
    corpus_name = [i.split("\t")[0] for i in data if i.split("\t")[1].strip() == code]
    path_map = {}
    zero_map = {}
    for name in corpus_name:
        for txt in corpus_file:
            if txt.startswith(name):
                # print(txt[0:-4].split("$")[1])
                if txt[0:-4].split("$")[1]!='0':
                    path_map[name] = "{}\\{}".format(dir_path,txt)
                else:
                    zero_map[name]="{}\\{}".format(dir_path,txt)
    print(zero_map)
    return path_map,zero_map


'''
调用get_path()方法
合并两个map 写入json文件
return: api_txt_map.json
'''
def get_map():
    malware_dir = r"G:\sln\malware_api"
    benign_dir = r"G:\sln\benign_api"
    malware_map,malware_zero_map = get_path(malware_dir,'1')
    benign_map,benign_zero_map = get_path(benign_dir,'0')
    dic = dict(malware_map,**benign_map)
    print(len(dic.keys()))
    utils.write_json("api_data/process/api_txt_map.json",dic)
    dic2 = dict(malware_zero_map, **benign_zero_map)
    print(len(dic2.keys()))
    utils.write_json("api_data/process/api_zero_txt_map.json",dic2)

def get_dataset_withoutzero():
    malware_dir = r"G:\sln\malware"
    benign_dir = r"G:\sln\benign"
    zero_api = utils.read_json("api_data/process/api_zero_txt_map.json")
    malware_list = os.listdir(malware_dir)
    benign_list = os.listdir(benign_dir)
    # print(malware_list)
    # print(benign_list)
    result = []
    for malware in malware_list:
        if malware not in zero_api.keys():
            result.append("{}\t{}".format(malware, 1))
    for benign in benign_list:
        if benign not in zero_api.keys():
            result.append("{}\t{}".format(benign, 0))
    print(len(result))
    utils.write_file("api_data/dataset.txt", result)

'''
将数据集中apk对应的api整合为一行
所有apk一共12696个 对应12696行api序列
return: apk_api_corpus.txt
'''
def get_corpus():
    name =[i.strip().split("\t")[0] for i in utils.read_file("api_data/dataset.txt")]
    path_map = utils.read_json("api_data/process/api_txt_map.json")
    print(len(name),len(path_map.keys()))
    result = []
    for index,n in enumerate(name):
        if (index+1) % 500 == 0:
            print("{} have finish!".format(index))
        with open(path_map[n],'r',encoding="utf8") as f:
            result.append(" ".join(f.read().split("\n")))
    utils.write_file(r"G:\sln\graphsage\corpus\apk_api_corpus.txt",result)

'''
计算每个api在对应类别（1/0）中出现的频率
frequency = count / N
'''
def calc_rt(apk_dir):
    file_name = os.listdir(apk_dir)
    path_map = utils.read_json("api_data/process/api_txt_map.json")
    N = len(path_map.keys())
    print(N,len(file_name))
    api_dict = defaultdict(int)
    for name in file_name:
        if name not in path_map.keys():
            continue
        api_list = [i.strip() for i in utils.read_file(path_map[name])]
        api_set = set(api_list)
        for api in api_set:
            api_dict[api] = api_dict[api] + 1
        # print(name,len(api_list),len(api_set))
    default_api_dict = defaultdict(int)
    for k,v in api_dict.items():
        default_api_dict[k] = v/N
    print(N, len(api_dict.items()))
    return default_api_dict


def calc_mbrt():
    mrt_api_dict = calc_rt(r"G:\sln\malware")
    brt_api_dict = calc_rt(r"G:\sln\benign")
    api_dict = mrt_api_dict.keys()|brt_api_dict.keys()
    print(len(api_dict))
    api_mbrt_dict = defaultdict(int)
    for api in api_dict:
        api_mbrt_dict[api] = (mrt_api_dict[api],brt_api_dict[api])
    utils.write_json("api_data/process/api_mbrt_dict.json",api_mbrt_dict)


def calc_epio():
    api_mbrt = utils.read_json("api_data/process/api_mbrt_dict.json")
    api_epio = {}
    for api in api_mbrt.keys():
        mrt,brt = api_mbrt[api]
        epio1 = math.log(1+mrt/(brt+0.001))
        epio2 = mrt/(mrt+brt)
        epio3 = mrt*math.log(1/(brt+0.001))
        api_epio[api] = (epio1,epio2,epio3)
        print("%20.20f %20.20f %20.20f %20s"%(epio1,epio2,epio3,api))
    utils.write_json("api_data/process/api_epio_dict.json",api_epio)


'''
解析smali时部分apk解析失败
会在目录下产生空文件夹 
'''
def find_null_dir():
    dir_path= r"G:\sln\benign"
    for file in os.listdir(dir_path):
        if len(os.listdir("{}/{}".format(dir_path,file))) == 0:
            print(file)
'''
'''
def find_null_api():
    dir_path = r"G:\sln\benign_api"
    null_api = []
    api = utils.read_line_list("api_data/dataset.txt")
    api = [i.strip().split("\t")[0] for i in api]
    for file in os.listdir(dir_path):
        if file.split("$")[1]=='0.txt':
            null_api.append(file.split("$")[0])
    print(null_api)
    print(api)
    index = []
    for ind,i in enumerate(api):
        if i in null_api:
            index.append(ind)
    logger = utils.Logging()
    logger.info(index)
    print(len(null_api))


def logtest():
    logger = utils.Logging()
    logger.info("hhhhhh")

'''
挖掘敏感api的使用模式
'''
def get_word2vec_corpus(dir_path,json_path):
    path_map = utils.read_json("api_data/process/api_txt_map.json")
    smali_dirs = os.listdir(dir_path)
    # print(smali_dirs)
    for ind,smali in enumerate(smali_dirs):

        sensitive_called_corpus = []
        # package = set()
        # for dirpath,dirs,files in os.walk("{}/{}/smali".format(dir_path,smali)):
        #     for filename in fnmatch.filter(files,'*.smali'):
        #         package.add("L{}".format(dirpath.split("smali")[1][1:].replace("\\","/")))
        call = utils.read_json("{}/{}.json".format(json_path,smali))
        print(ind,call['apk_name'],call['method_number'])
        sensitive_api = utils.read_line_enter(path_map[smali])
        print(len(sensitive_api))
        for method in call['parse_result']:
            called_lst = method['calling_to']
            if len(called_lst) ==0:
                pass
            else:
                for called in called_lst:
                    # 该method调用了敏感api
                    if called in sensitive_api:
                        # 作为一个corpus加入语料库
                        sensitive_called_corpus.append(" ".join(called_lst))
        utils.write_line_list(r"G:\sln\corpus\api_api_corpus.txt", "a",sensitive_called_corpus)
        # print(len(sensitive_called_corpus))
        # print(sensitive_called_corpus)
        # delete_inner_api(sensitive_called_corpus,package)

def delete_inner_api(sensitive_called_corpus,package):
    clean_called_corpus = []
    for sensitive_called_apis in sensitive_called_corpus:
        clean_called_apis = []
        for sensitive_api in sensitive_called_apis:
            if sensitive_api[0:sensitive_api.rfind("/")] not in package:
                clean_called_apis.append(sensitive_api)
        if " ".join(clean_called_apis) == '':
            print(sensitive_called_apis)
        clean_called_corpus.append(" ".join(clean_called_apis))

    utils.write_line_list("api_data/process/test.txt",clean_called_corpus)


def is_inner(api,package):
    api = api[0:api.rfind("/")]
    return api in package


def get_word2vec():
    # corpus = utils.read_line_enter("G:/sln/corpus/sensitive_api_api_corpus.txt")
    # print(len(corpus))

    # sentences = LineSentence(open("G:\\sln\\corpus\\sensitive_api_api_corpus.txt",'r',encoding="utf8"))
    # model = gensim.models.Word2Vec(sentences=sentences,sg=1, vector_size=50, min_count=1, window=3,compute_loss=True, seed=100)
    # print(model.get_latest_training_loss())
    # model.save("G:\\sln\\corpus\\sensitive_api_api_corpus.model")


    model = KeyedVectors.load("G:\\sln\\corpus\\sensitive_api_api_corpus.model")
    print(model.wv.vectors.shape)
    # print(model.wv.key_to_index)
    v1 = model.wv['Landroid/accounts/AccountManager->addAccount']
    v2 = model.wv["Landroid/widget/Button-><init>"]
    v3 = model.wv['Landroid/accounts/AccountManager->getAccountsByType']
    # print(1/utils.get_dist_similar(v1,v2))
    # print(1 /utils.get_dist_similar(v1, v3))
    print(utils.get_cos_similar(v1,v2))
    print(utils.get_cos_similar(v1,v3))

    # print(model.wv.similarity('Landroid/accounts/AccountManager->addAccount','Landroid/accounts/AccountManager->getAccountsByType'))
    # print(model.wv.similarity('Landroid/accounts/AccountManager->addAccount','Landroid/app/LocalActivityManager->dispatchStop'))

'''
出现在同一method中的敏感API
'''
def get_method_sensitive(dir_path_lst):
    # sensitive_api = utils.read_line_enter("api_data/process/transSenApiAll.txt")
    path_map = defaultdict(str, utils.read_json("api_data/process/api_txt_map.json"))
    pattern = {}
    for dir_path in dir_path_lst:
        for file in tqdm(os.listdir(dir_path), desc=dir_path):
            if not os.path.exists(path_map[file[0:-5]]):
                print(file)
                continue;
            sensitive_api = set(utils.read_line_enter(path_map[file[0:-5]]))
            call = utils.read_json("{}/{}".format(dir_path, file))
            apk_pattern = []
            for method in call['parse_result']:
                method_sensitive_set = set()
                called_lst = method['calling_to']
                for called in called_lst:
                    if called in sensitive_api:
                        method_sensitive_set.add(called)
                if len(method_sensitive_set) > 1:
                    apk_pattern.append(list(method_sensitive_set))
            pattern[file[0:-5]] = apk_pattern
    utils.write_json("G:/sln/corpus/simple_pattern.json", pattern)


def get_class_method(call):
    class_method_map = defaultdict(list)
    for method in call['parse_result']:
        call_class = method['id'].split('->')[0]
        class_method_map[call_class].append(method['id'])
    return class_method_map

'''
出现在同一class中的敏感API
'''
def get_class_sensitive(dir_path_lst):
    # sensitive_api = utils.read_line_enter("api_data/process/transSenApiAll.txt")
    path_map = defaultdict(str, utils.read_json("api_data/process/api_txt_map.json"))
    pattern = {}
    for dir_path in dir_path_lst:
        for file in tqdm(os.listdir(dir_path), desc=dir_path):
            if not os.path.exists(path_map[file[0:-5]]):
                print(file)
                continue;
            sensitive_api = set(utils.read_line_enter(path_map[file[0:-5]]))
            call = utils.read_json("{}/{}".format(dir_path, file))
            class_method_map = get_class_method(call)
            apk_pattern = {}
            for method in call['parse_result']:
                method_sensitive_set = set()
                called_lst = method['calling_to']
                for called in called_lst:
                    if called in sensitive_api:
                        method_sensitive_set.add(called)
                if len(method_sensitive_set) >= 1:
                    list(method_sensitive_set).sort()
                    apk_pattern[method['id']]=list(method_sensitive_set)
            apk_class_pattern = defaultdict(dict)
            for class_name in class_method_map.keys():
                for method_name in class_method_map[class_name]:
                    if method_name in apk_pattern:
                        apk_class_pattern[class_name][method_name]= apk_pattern[method_name]
            pattern[file[0:-5]] = apk_class_pattern
    utils.write_json("G:/sln/corpus/class_simple_pattern.json", pattern)


def get_pattern_corpus():
    dir_path_lst = [r"G:/sln/malware_extract",r"G:/sln/benign_extract"]
    # dir_path_lst = [r"G:/sln/test_extract"]
    # sensitive_api = utils.read_line_enter("api_data/process/transSenApiAll.txt")
    path_map = defaultdict(str, utils.read_json("api_data/process/api_txt_map.json"))
    pattern = {}
    for dir_path in dir_path_lst:
        for file in tqdm(os.listdir(dir_path), desc=dir_path):
            if not os.path.exists(path_map[file[0:-5]]):
                print(file)
                continue;
            sensitive_api = set(utils.read_line_enter(path_map[file[0:-5]]))
            call = utils.read_json("{}/{}".format(dir_path, file))
            apk_pattern = []
            for method in call['parse_result']:
                method_sensitive_set = set()
                called_lst = method['calling_to']
                for called in called_lst:
                    if called in sensitive_api:
                        method_sensitive_set.add(called)
                if len(method_sensitive_set) > 0:
                    apk_pattern.append(list(method_sensitive_set))
            pattern[file[0:-5]] = apk_pattern
    # print(pattern)
    # print(len(pattern.keys()))
    corpus = []
    for k in tqdm(pattern.keys()):
        para = []
        apk = pattern[k]
        for method in range(len(apk)):
            para.append(' '.join(apk[method]))
        corpus.append(' '.join(para))
    utils.write_line_list("G:/sln/corpus/para_sensitive_api_api_corpus.txt",'w',corpus)


def get_pattern_pairs():
    # pattern = utils.read_json("G:/sln/corpus/simple_pattern.json")
    pattern = utils.read_json("G:/sln/corpus/class_pattern.json")
    print(len(pattern.keys()))
    pattern_set = set()
    for k in pattern.keys():
        apk = pattern[k]
        for method in range(len(apk)):
            apk[method].sort()
            for i in range(len(apk[method])):
                for j in range(i+1,len(apk[method])):
                    pattern_set.add(apk[method][i]+' '+apk[method][j])
    pattern_lst = list(pattern_set)
    pattern_lst.sort()
    # utils.write_line_list("api_data/process/api_pairs.txt",'w',pattern_lst)
    utils.write_line_list("api_data/process/class_api_pairs.txt",'w',pattern_lst)

def calc_cocurrence():
    dir_path = "G:\\sln\\corpus"
    model_name_lst = ['api_api_corpus.model','sensitive_api_api_corpus.model','para_sensitive_api_api_corpus.model']
    # api_pairs = utils.read_line_enter("api_data/process/api_pairs.txt")
    api_pairs = utils.read_line_enter("api_data/process/class_api_pairs.txt")

    model_dict = {}
    for model_path in model_name_lst:
        model = KeyedVectors.load("{}\\{}".format(dir_path, model_path))
        type = model_path.split("_")[0]
        model_dict[type] = model

    api_pairs_cocorrence = []

    for pair in tqdm(api_pairs):
        pair_calc = {}
        pair = pair.split(" ")
        pair_calc['pair'] = pair
        for k in model_dict.keys():
            model = model_dict[k]
            cocurr = utils.get_cos_similar(model.wv[pair[0]], model.wv[pair[1]])
            pair_calc['cos_' + k] = cocurr
        api_pairs_cocorrence.append(pair_calc)
    print(len(api_pairs_cocorrence))
    utils.write_json('api_data/process/class_api_pairs_cocorrence.json',{'api_pairs_cocurrence':api_pairs_cocorrence})


def test():
    # t = utils.read_json("G:/sln/corpus/class_simple_pattern.json")
    # pattern = defaultdict(list)
    # for k in t.keys():
    #     for c in t[k].keys():
    #         test_list = []
    #         for x in t[k][c].values():
    #             test_list +=x
    #         if len(test_list) > 1:
    #             pattern[k].append(test_list)
    # utils.write_json("G:/sln/corpus/class_pattern.json",pattern)
    a = utils.read_json("G:/sln/corpus/class_pattern.json")
    print(len(a.keys()))


if __name__ == '__main__':
    # get_dataset_withoutzero()
    # get_label_data()
    # split()
    # get_map()
    # get_corpus()
    # find_null_dir()
    # calc_mbrt()
    calc_epio()
    # find_null_api()
    # logtest()
    # get_word2vec_corpus("G:/sln/benign","G:/sln/benign_extract")
    # get_word2vec()
    # get_method_sensitive([r"G:\sln\malware_extract",r"G:\sln\benign_extract"])
    # get_pattern_pairs()
    # get_pattern_corpus()
    # calc_cocurrence()
    # get_class_method(utils.read_json(r"G:\sln\malware_extract\0b4bd8dc61ab8df5a42b3c1d83d2821a.json"))
    # get_class_sensitive([r"G:\sln\malware_extract",r"G:\sln\benign_extract"])
    # te_simst()