from math import log
import numpy as np
import os

# 预处理，VSM.py和dataset文件夹放在同一目录下
file_list = os.listdir("dataset/")
# 文档列表
docs = []
# 词项集合
term_set = set()

for file in file_list:
    doc = open(os.path.join("dataset/",file)).read()
    doc = doc.replace("\n", " ")
    doc = doc.replace(":", "")
    doc = doc.replace(",", "")
    doc = doc.replace(".", "")
    doc = doc.replace(";", "")
    doc = doc.replace("?", "")
    doc = doc.replace("!", "")
    docs.append(doc.lower())

for doc in docs:
    split = doc.split(' ')
    term_set = term_set.union(set(split))

# 计算TF
## 文档的tf向量词典
TF_dicts = dict.fromkeys(docs,{})
# 求每篇文档的tf向量
for doc in docs:
    tf_N = dict.fromkeys(term_set, 0)
    TF_dict = dict.fromkeys(term_set, 0)
    split = doc.split(' ')
    for word in split:
        tf_N[word] += 1
    for term in  tf_N.keys():
        TF_dict[term] = log(tf_N[term] + 1, 10)
    #print('TF_dict: ', TF_dict)
    TF_dicts[doc] = TF_dict

# 求每个词项的idf
IDF_dict = dict.fromkeys(term_set,0)
for term in term_set:
    df = 0
    for doc in docs:
        if term in doc:
            df += 1
    IDF_dict[term] = log((docs.__len__()/df), 10)
    #print('term\'s idf: ', IDF_dict[term])

# 文档向量
doc_vectors = []
for doc in docs:
    TF_dict = TF_dicts[doc]
    doc_vec = []
    for term in sorted(TF_dict.keys()):
        doc_vec.append(TF_dict[term]*IDF_dict[term])
    doc_vectors.append(doc_vec)

# 文档向量的模长
mod_dict = []
for doc_vec in doc_vectors:
    mod_dict.append(np.linalg.norm(np.array(doc_vec)))



# 查询向量

#从键盘输入，或直接指定查询向量（如q = "sky and the"）
q = input("请输入查询向量：")
split_q = q.split(' ')

# q的tf
q_tf_dict = dict.fromkeys(term_set, 0)
q_tf_N = dict.fromkeys(term_set, 0)
for word in split_q:
    q_tf_N[word] += 1
for term in q_tf_N.keys():
    q_tf_dict[term] = log(q_tf_N[term] + 1, 10)

# q的vector
q_vec = []
for term in sorted(q_tf_dict.keys()):
    q_vec.append(q_tf_dict[term]*IDF_dict[term])

# 计算相似度
j = 0
for doc_vec in doc_vectors:
    res_sum = 0
    for i in range(len(doc_vec)):
        res_sum += q_vec[i]*doc_vec[i]
    res_cos = res_sum/mod_dict[j]
    j += 1
    print(f'Q and Doc{j}\'s similarity:', res_cos)










        

    


