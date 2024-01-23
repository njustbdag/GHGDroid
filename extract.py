import json
import re
import os
import fnmatch
import time
import math
import threading
import concurrent.futures


pattern_class_name = re.compile(r'^\.class.*\ (.+(?=\;))', re.MULTILINE)
pattern_method_data = re.compile(r'^\.method.+?\ (.+?(?=\())\((.*?)\)(.*?$)(.*?(?=\.end\ method))',
                                      re.MULTILINE | re.DOTALL)
pattern_called_methods = re.compile(
    r'invoke-.*?\ {(.*?)}, (.+?(?=;))\;\-\>(.+?(?=\())\((.*?)\)(.*?)(?=$|;)', re.MULTILINE | re.DOTALL)
pattern_move_result = re.compile(r'move-result.+?(.*?)$', re.MULTILINE | re.DOTALL)


def parse_smali_files(dir):
    result = []
    """
    parses all smali files in the
    :return: null
    """
    # print ('Parsing smali files {}'.format(dir))
    all_file_list= []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*.smali'):
            filepath = os.path.join(root, filename)
            all_file_list.append(filepath)
    if len(all_file_list) == 0:
        return result
    n = int(math.ceil(len(all_file_list))/float(20))
    threads = []
    for i in range(0,len(all_file_list),n):
        file_list = all_file_list[i:i+n]
        threads.append(threading.Thread(target=parse_smali_files_thread,args=(file_list,result)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("multi_thread end")
    return result


def parse_smali_files_thread(file_list,result):

    for filepath  in file_list:
        with open('\\\?\\'+filepath, 'r',encoding="utf8") as smali_file:
            content = smali_file.read()
            parse_smali_file(content,result)

def parse_smali_file(content,result):
    """
    parses a single smali file and inserts the data to the smali database
    :param content: content of the smali file
    :return: null
    """
    # TODO add the regex to parse the called function and change the FOR loop
    # TODO change code to use all the new methods
    # parse class
    class_name = get_class_name(content)
    methods = get_methods(content)
    # add methods to db
    for method in methods:
        method_name = method[0].split(' ')[-1]
        method_data = method[3]
        if class_name.startswith("LAndroid"):
            class_name = "La" + class_name[2:]
        method_id = '%s->%s' % (class_name, method_name)
        called_methods = get_called_methods(method_data)
        # method_calling_to = ''
        method_calling_to = []

        for called_method in called_methods:
            # method_calling_to = '%s%s->%s,' % (method_calling_to, called_method[1], called_method[2])
            if (called_method[1].startswith("LAnd")):
                c1 = "La" + called_method[1][2:]
            else:
                c1 = called_method[1]
            called_mt = '%s->%s' % (c1, called_method[2])
            method_calling_to.append(called_mt)
        dicc = {"id": method_id,"calling_to": method_calling_to}
        result.append(dicc)

def get_class_name(content):
    """
    gets the class name of a single smali file content
    :param content: smali file content
    :rtype: string
    :return: the name of the class
    """
    data = re.findall(pattern_class_name, content)
    return data[0]

def get_methods(content):
    """
    gets all methods in a single smali file content
    :param content: smali file content
    :rtype: list of lists
    :return: [0] - method name
             [1] - method parameters
             [2] - method return value
             [3] - method data
    """
    data = re.findall(pattern_method_data, content)
    return data

def get_called_methods(content):
    """
    gets all the method called inside a smali method data. works just fine with a single smali line
    :param content: content of the smali data to be parsed
    :rtype: list of lists
    :return: [0] - called method parameters
             [1] - called method object type
             [2] - called method name
             [3] - called method parameters object type
             [4] - called method return object type
    """
    data = re.findall(pattern_called_methods, content)
    return data


def batch_parser(input_dir,output_dir):
    smali_dirs = os.listdir(input_dir)[0:20]
    for smali_dir in smali_dirs:
        if os.path.exists("{}/{}.json".format(output_dir,smali_dir)):
            print("{}已存在解析产物".format(smali_dir))
        else:
            start = time.time()
            result = parse_smali_files(r"{}\{}\smali".format(input_dir,smali_dir))
            end = time.time()
            extract_time = end - start
            extract_info = {"apk_name":smali_dir,
                         "method_number":len(result),
                         "extract_time":extract_time,
                         "parse_result":result,
                         }
            json_data = json.dumps(extract_info)
            out_json = r"{}\{}.json".format(output_dir,smali_dir)
            with open(out_json, 'w',encoding="utf8") as f:
                f.write(json_data)
            print(smali_dir+" finish!!!")



if __name__== '__main__':
    start = time.time()
    input_path = r"E:\sln\malware"
    output_path = r"E:\sln\malware_extract"
    batch_parser(input_path,output_path)
    end = time.time()
    print(end-start)
    # input_path = r"E:\sln\benign"
    # output_path = r"E:\sln\benign_extract"
    # batch_parser(input_path, output_path)

