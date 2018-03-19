# coding: utf-8

import json
from nltk.tokenize import word_tokenize
from gensim.summarization.bm25 import BM25
from tqdm import tqdm
import numpy as np


data = json.load(open('./alice.json'))


all_data = data
all_data = [x for x in all_data if '???' not in x[0]]


docs = [
    word_tokenize(x[0])
    for x in tqdm(all_data)
]

bm25 = BM25(docs)

average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

while True:
    print('=' * 30)
    s = input('your question:')
    s = s.strip()
    if len(s) <= 0:
        print('too short')
        continue

    query_doc = word_tokenize(s)
    scores = bm25.get_scores(query_doc, average_idf)
    for n in reversed(np.argsort(scores)[-5:]):
        print(all_data[n][0])
        print(all_data[n][1])
        print(scores[n])
        print('-' * 30)
