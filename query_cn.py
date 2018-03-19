# coding: utf-8

import json
from gensim.summarization.bm25 import BM25
from tqdm import tqdm
import numpy as np
# import jieba

data = json.load(open('./alice_zh_gg.json'))
data2 = json.load(open('./alice_zh_bd.json'))


all_data = data + data2
all_data = [x for x in all_data if '??' not in x[0]]


docs = [
    list(x[2]) # jieba.lcut(x[2])
    for x in tqdm(all_data)
]

bm25 = BM25(docs)

average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

while True:
    print('=' * 30)
    print('=' * 30)
    s = input('your question:')
    print('=' * 30)
    s = s.strip()
    if len(s) <= 0:
        print('too short')
        continue

    query_doc = list(s) # jieba.lcut('你喜欢我吗')
    scores = bm25.get_scores(query_doc, average_idf)
    for n in reversed(np.argsort(scores)[-5:]):
        print(all_data[n][2])
        print(all_data[n][3])
        print(scores[n])
        print('-' * 30)
