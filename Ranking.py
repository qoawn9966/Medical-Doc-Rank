#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# simple similarity search on FAQ

import numpy as np
from bert_serving.client import BertClient
from termcolor import colored
import csv
import TFIDF
import sys
import os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from rank_bm25 import BM25Okapi, BM25L, BM25Plus

import pandas as pd
from pandas import DataFrame

prefix_a = '**A:**  '
prefix_n = '**N:** '
prefix_t = '##### **Q:** '
topk = 100




def Rank() :
    """with open(path, encoding='cp949') as fp:
    questions = [v.replace(prefix_a, '').strip() for v in fp if v.strip() and v.startswith(prefix_a)]
    with open(path, encoding='cp949') as fp:
        NCT_ID = [v.replace(prefix_n, '').strip() for v in fp if v.strip() and v.startswith(prefix_n)]
        with open(path, encoding='cp949') as fp:
            Title = [v.replace(prefix_t, '').strip() for v in fp if v.strip() and v.startswith(prefix_t)]

    print(questions)
    print(NCT_ID)
    print('%d papers loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))
"""

with BertClient(port=5555, port_out=5556) as bc:

    #df = pd.read_csv('D:\#2. Personal Study\TREC\Data Set/2018 Precision Medicine Track/TSV/4_Melanoma.csv', encoding='cp949')
    df_1 = pd.read_csv('D:\#2. Personal Study\TREC\Data Set/2018 Precision Medicine Track/TSV/1_Breast Cancer.csv', encoding='cp949')
    df_2 = pd.read_csv('D:\#2. Personal Study\TREC\Data Set/2018 Precision Medicine Track/TSV/2_Healthy.csv', encoding='cp949')
    df_3 = pd.read_csv('D:\#2. Personal Study\TREC\Data Set/2018 Precision Medicine Track/TSV/3_HIV.csv', encoding='cp949')
    df_4 = pd.read_csv('D:\#2. Personal Study\TREC\Data Set/2018 Precision Medicine Track/TSV/4_Melanoma.csv', encoding='cp949')
    df_5 = pd.read_csv('D:\#2. Personal Study\TREC\Data Set/2018 Precision Medicine Track/TSV/5_Prostate Cancer.csv', encoding='cp949')

    allData = []
    allData.append(df_1)
    allData.append(df_2)
    allData.append(df_3)
    allData.append(df_4)
    allData.append(df_5)
    print(allData)



    #df = pd.concat(allData, axis=0, ignore_index=True, sort=False)
    df = pd.concat([df_1, df_2, df_3, df_4, df_5], ignore_index=True)

    print(df)
    df.to_csv()


    Title = df['Title']
    Abstract = df['Abstract']
    NCT_ID = df['NCT_ID']

    #print(questions)

    doc_vecs_abstract = bc.encode(df.Abstract.tolist())
    doc_vecs_tiltle = bc.encode(df.Title.tolist())


    tokenized_corpus_Title = [doc.split(" ") for doc in Title]
    tokenized_corpus_Abstract = [doc.split(" ") for doc in Abstract]

    bm25_Title = BM25Okapi(tokenized_corpus_Title)
    bm25_Abstract = BM25Okapi(tokenized_corpus_Abstract)
    n = 100

    while True:
        print("start")
        Topic = input(colored('Topic: ', 'magenta'))
        query = input(colored('your question: ', 'green'))
        gene = input(colored('Gene: ', 'green'))
        if query =='out':
            break
        query_vec = bc.encode([query])[0]

        # compute BM25 as score
        tokenized_query = gene.split(" ")
        BM25_Score_Title = bm25_Title.get_scores(tokenized_query) *2
        BM25_Score_Abstract = bm25_Abstract.get_scores(tokenized_query) *2
        BM25_Score = BM25_Score_Title + BM25_Score_Abstract

        # compute normalized dot product as score
        score_abstract = np.sum(query_vec * doc_vecs_abstract, axis=1) / np.linalg.norm(doc_vecs_abstract, axis=1)
        score_title = np.sum(query_vec * doc_vecs_tiltle, axis=1) / np.linalg.norm(doc_vecs_tiltle, axis=1)

        # compute TF-IDF as score
        # tfidf = TFIDF.tfIDF(Abstract)
        #score_TFIDF = tfidf.search(gene)
        #print(score_TFIDF)

        #score = np.sum(query_vec * doc_vecs_abstract, axis=1) / np.linalg.norm(doc_vecs_abstract, axis=1)
        #topk_idx = np.argsort(score_abstract + score_title + score_TFIDF)[::-1][:topk]
        #topk_idx = np.argsort(score_abstract + score_title)[::-1][:topk]
        topk_idx = np.argsort(score_abstract + score_title + BM25_Score)[::-1][:topk]

        print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))

        for idx in topk_idx:
            with open('D:\#2. Personal Study\TREC\Data Set\\2018 Precision Medicine Track\Ranking Data/Ranking_CWE_BM25.csv', 'a', newline='') as csvfile:
                fieldnames = ['QueryNum', 'Q0', 'NCT_ID', 'Rank', 'Score', 'STANDARD']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                #score = score_abstract[idx] + score_title[idx]
                #score = score_abstract[idx] + score_title[idx] + score_TFIDF[idx]
                score = score_abstract[idx] + score_title[idx] + BM25_Score[idx]
                #score = BM25_Score[idx]
                print('> %s\t%s' % (colored('%.1f' % score, 'cyan'), colored(Title[idx], 'yellow')))
                #print("tfidf: ", score_TFIDF[idx])
                print("BM25: ", BM25_Score[idx])
                print("CWE: ", score_abstract[idx] + score_title[idx])
                print(NCT_ID[idx])
                writer.writerow({'QueryNum': Topic, 'Q0': 'Q0', 'NCT_ID': NCT_ID[idx], 'Rank': '1', 'Score': score, 'STANDARD': 'STANDARD'})

                #csvfile.writerow({questions[idx], str(score[idx])})
                #csvfile.write(str(score[idx]))



                #RankingData = [[questions[idx], score[idx]]]

                #df = pd.DataFrame(RankingData)

            #df.loc[idx] = pd.DataFrame(RankingData)
        #df.to_csv('D:\#2. Personal Study\TREC\Data Set\Ranking Data/Ranking.csv')

Rank()

"""
while True:
    Disease = input(colored('Classified Disease: ', 'magenta'))
    if Disease== 'Breast Cancer':
        Rank('D:/#2. Personal Study/bert-as-service-master/bert-as-service-master/Breast Cancer.md')
    if Disease== 'Prostate Cancer':
        Rank('D:/#2. Personal Study/bert-as-service-master/bert-as-service-master/Prostate Cancer.md')
    if Disease== 'Healthy':
        Rank('D:/#2. Personal Study/bert-as-service-master/bert-as-service-master/Healthy.md')
    if Disease== 'HIV':
        Rank()
    if Disease == 'all':
        Rank('D:\#2. Personal Study\\bert-as-service-master\\bert-as-service-master/All_Data.md')
    if Disease == 'melanoma':
        Rank('D:\#2. Personal Study\\bert-as-service-master\\bert-as-service-master/Melanoma.md')
    if Disease== 'out':
        break
"""