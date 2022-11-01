import operator
from math import log
import numpy as np
import os

# 预处理，VSM.py和dataset文件夹放在同一目录下
file_list = os.listdir("dataset/")

#####################################################################################################
# 诗歌名称构建向量：
#####################################################################################################
# 文档列表
title_docs = []
# 词项集合
title_term_set = set()
for file in file_list:
    doc = file.replace(".txt","")
    title_docs.append(doc.lower())

for doc in title_docs:
    split = doc.split(' ')
    title_term_set = title_term_set.union(set(split))

# 计算TF
## 文档的tf向量词典
title_TF_dicts = dict.fromkeys(title_docs,{})
# 求每篇文档的tf向量
for doc in title_docs:
    tf_N = dict.fromkeys(title_term_set, 0)
    TF_dict = dict.fromkeys(title_term_set, 0)
    split = doc.split(' ')
    for word in split:
        tf_N[word] += 1
    for term in  tf_N.keys():
        TF_dict[term] = log(tf_N[term] + 1, 10)
    #print('TF_dict: ', TF_dict)
    title_TF_dicts[doc] = TF_dict

# 求每个词项的idf
title_IDF_dict = dict.fromkeys(title_term_set,0)
for term in title_term_set:
    df = 0
    for doc in title_docs:
        if term in doc:
            df += 1
    title_IDF_dict[term] = log((title_docs.__len__()/df), 10)
    #print('term\'s idf: ', title_IDF_dict[term])

# 文档向量
title_doc_vectors = []
for doc in title_docs:
    TF_dict = title_TF_dicts[doc]
    doc_vec = []
    for term in sorted(TF_dict.keys()):
        doc_vec.append(TF_dict[term]*title_IDF_dict[term])
    title_doc_vectors.append(doc_vec)

# 文档向量的模长
title_mod_dict = []
for doc_vec in title_doc_vectors:
    title_mod_dict.append(np.linalg.norm(np.array(doc_vec)))

#####################################################################################################
# 作者构建文档向量：
#####################################################################################################
# 文档列表
author_docs = []
# 词项集合
author_term_set = set()
for file in file_list:
    doc = open(os.path.join("dataset/",file)).readline() # 只读第一行的作者
    doc = doc[8:] # 去掉"Author: "
    doc = doc.replace("\n", "")
    author_docs.append(doc.lower())
# 去重
#author_docs = list(set(author_docs))

for doc in author_docs:
    split = doc.split(' ')
    author_term_set = author_term_set.union(set(split))

# 计算TF
## 文档的tf向量词典
author_TF_dicts = dict.fromkeys(author_docs,{})
# 求每篇文档的tf向量
for doc in author_docs:
    tf_N = dict.fromkeys(author_term_set, 0)
    TF_dict = dict.fromkeys(author_term_set, 0)
    split = doc.split(' ')
    for word in split:
        tf_N[word] += 1
    for term in  tf_N.keys():
        TF_dict[term] = log(tf_N[term] + 1, 10)
    #print('TF_dict: ', TF_dict)
    author_TF_dicts[doc] = TF_dict

# 求每个词项的idf
author_IDF_dict = dict.fromkeys(author_term_set,0)
for term in author_term_set:
    df = 0
    for doc in author_docs:
        if term in doc:
            df += 1
    author_IDF_dict[term] = log((author_docs.__len__()/df), 10)
    #print('term\'s idf: ', IDF_dict[term])

# 文档向量
author_doc_vectors = []
for doc in author_docs:
    TF_dict = author_TF_dicts[doc]
    doc_vec = []
    for term in sorted(TF_dict.keys()):
        doc_vec.append(TF_dict[term]*author_IDF_dict[term])
    author_doc_vectors.append(doc_vec)

# 文档向量的模长
author_mod_dict = []
for doc_vec in author_doc_vectors:
    author_mod_dict.append(np.linalg.norm(np.array(doc_vec)))

#####################################################################################################
# 诗歌内容构建向量：
#####################################################################################################
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


#####################################################################################################
# 文档查询向量、诗歌名查询向量、作者查询向量
#####################################################################################################
#从键盘输入
# 诗歌名查询向量
titleq = input("请输入查询诗歌名关键字：")
titleq.lower()
split_titleq = titleq.split(' ')

# titleq的tf
titleq_tf_dict = dict.fromkeys(title_term_set, 0)
titleq_tf_N = dict.fromkeys(title_term_set, 0)
for word in split_titleq:
    if word in title_term_set:
        titleq_tf_N[word] += 1
for term in titleq_tf_N.keys():
    titleq_tf_dict[term] = log(titleq_tf_N[term] + 1, 10)

# titleq的vector
titleq_vec = []
for term in sorted(titleq_tf_dict.keys()):
    titleq_vec.append(titleq_tf_dict[term]*title_IDF_dict[term])

# 作者查询向量
authorq = input("请输入查询作者名：")
authorq.lower()
split_authorq = authorq.split(' ')
# authorq的tf
authorq_tf_dict = dict.fromkeys(author_term_set, 0)
authorq_tf_N = dict.fromkeys(author_term_set, 0)
for word in split_authorq:
    if word in author_term_set:
        authorq_tf_N[word] += 1
for term in authorq_tf_N.keys():
    authorq_tf_dict[term] = log(authorq_tf_N[term] + 1, 10)

# q的vector
authorq_vec = []
for term in sorted(authorq_tf_dict.keys()):
    authorq_vec.append(authorq_tf_dict[term]*author_IDF_dict[term])

#文档查询向量
q = input("请输入查询文档内容关键字：")
q.lower()
split_q = q.split(' ')

# q的tf
q_tf_dict = dict.fromkeys(term_set, 0)
q_tf_N = dict.fromkeys(term_set, 0)
for word in split_q:
    if word in term_set:
        q_tf_N[word] += 1
for term in q_tf_N.keys():
    q_tf_dict[term] = log(q_tf_N[term] + 1, 10)

# q的vector
q_vec = []
for term in sorted(q_tf_dict.keys()):
    q_vec.append(q_tf_dict[term]*IDF_dict[term])

#####################################################################################################
# 计算相似度
#####################################################################################################
j = 0
title_rescos_list = []
for title_doc_vec in title_doc_vectors:
    res_sum = 0
    for i in range(len(title_doc_vec)):
        res_sum += titleq_vec[i]*title_doc_vec[i]
    res_cos = res_sum/title_mod_dict[j]
    title_rescos_list.append(res_cos)
    j += 1
    print(f'TITLEQ and Doc{j}\'s similarity:', res_cos)
#title_rescos_list = sorted(title_rescos_list)
print("title_rescos_list", title_rescos_list)
print("=======================================================================")

j = 0
author_rescos_list = []
for author_doc_vec in author_doc_vectors:
    res_sum = 0
    for i in range(len(author_doc_vec)):
        res_sum += authorq_vec[i]*author_doc_vec[i]
    res_cos = res_sum/author_mod_dict[j]
    author_rescos_list.append(res_cos)
    j += 1
    print(f'AUTHORQ and Doc{j}\'s similarity:', res_cos)
#author_rescos_list = sorted(author_rescos_list)
print("author_rescos_list", author_rescos_list)
print("=======================================================================")

j = 0
rescos_list = []
for doc_vec in doc_vectors:
    res_sum = 0
    for i in range(len(doc_vec)):
        res_sum += q_vec[i]*doc_vec[i]
    res_cos = res_sum/mod_dict[j]
    rescos_list.append(res_cos)
    j += 1
    print(f'Q and Doc{j}\'s similarity:', res_cos)
#rescos_list = sorted(rescos_list)
print("rescos_list", rescos_list)
print("=======================================================================")

#####################################################################################################
# 返回查询结果：title > author > content排序
#####################################################################################################
num = file_list.__len__()
keys = [i for i in range(1,num+1)]
title_ret = dict.fromkeys(keys, 0)
for i in range(1, num + 1):
    title_ret[i] = title_rescos_list[i-1]

author_ret = dict.fromkeys(keys, 0)
for i in range(1, num + 1):
    author_ret[i] = author_rescos_list[i-1]

content_ret = dict.fromkeys(keys, 0)
for i in range(1, num + 1):
    content_ret[i] = rescos_list[i-1]

# 构建一个综合结果的向量
ret_vec = dict.fromkeys(keys, 0)
# 给一个权 wt > wa > wc
wt, wa, wc = 0.5, 0.3, 0.2
for i in range(1, num + 1):
    ret_vec[i] = title_ret[i]*wt + author_ret[i]*wa + content_ret[i]*wc
# 按照value值降序
sorted_res = dict(sorted(ret_vec.items(), key=operator.itemgetter(1), reverse=True))
print("results here: ")
print(sorted_res)


# sorted_title_ret = dict(sorted(title_ret.items(), key=operator.itemgetter(1), reverse=True))
# sorted_author_ret = dict(sorted(author_ret.items(), key=operator.itemgetter(1), reverse=True))
# sorted_content_ret = dict(sorted(content_ret.items(), key=operator.itemgetter(1), reverse=True))

# print("title res: ", sorted_title_ret)
# print("author res: ", sorted_author_ret)
# print("content res: ", sorted_content_ret)


        

    


