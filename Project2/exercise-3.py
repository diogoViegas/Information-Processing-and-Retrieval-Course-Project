from xml.dom import minidom

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import re
import os
import json
import math
import numpy as np
import pandas as pd
from statistics import mean
import networkx as nx
import collections
from itertools import combinations
from collections import OrderedDict


ps = PorterStemmer()

use_candidateLen = 1    # 0 - nao usa o tamanho do candidato || 1 - usa tamanha do candidato (#caracteres)
r_usebm25 = 1           # 0 - nao usa BM25 no RRFScore || 1 - usa BM25 no RRFScore
r_usetfidf = 0          # 0 - nao usa TF-IDF no RRFScore || 1 - usa TF-IDF no RRFScore
r_usetf = 0             # 0 - nao usa term frequency no RRFScore || 1 - usa term frequency no RRFScore
r_uselen = 0            # 0 - nao usa len do candidato no RRFScore || 1 - usa len do candidato no RRFScore
r_useng = 1             # 0 - nao usa numero de palavras do candidato no RRFScore || 1 - usa numero de palavras do candidato no RRFScore
r_usepr = 1             # 0 - nao usa PageRank no RRFScore || 1 - usa PageRank no RRFScore


def countGram(candidate):
    count = 0
    for c in candidate:
        if c == " ":
            count += 1

    return count + 1


def flatten(list):
    flat_list = []
    for subslist in list:
        for item in subslist:
            flat_list.append(item)
    return flat_list


def removeStopWords(list):
    docs = []
    stopWordList = set(stopwords.words("english"))
    for sentence in list:
        sentence = sentence.split()
        frase = []
        for word in sentence:
            if word.lower() not in stopWordList:
                frase.append(word)
            fraseAux = ' '.join(frase)
        docs.append(fraseAux)
    return docs


def preProcessing(list):
    docs = []
    for l in list:
        l = l.lower()  # put lower key
        l = re.sub("(.)*@(.)*", '', l)  # remove emails
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)|`|\"|$|_|\'|\-|`]", '', l)  # remove pontuation
        l = re.sub("\d+", '', l)  # remove numbers
        docs.append(str(l))
    return docs


def readXML():
    docs = {}
    i = 0
    docname = []
    for filename in os.listdir("train"):
        i += 1
        s = ''
        mydoc = minidom.parse("train/" + filename)
        words = mydoc.getElementsByTagName('word')
        filename_X = ''
        for c in filename:
            if c == '.':
                break
            filename_X += str(c)
        docname.append(filename_X)

        for word in words:
            s += ps.stem(word.firstChild.data)  # stemming
            s += ' '

        docs[filename] = s

    return docs, docname


def readJson():
    with open('test.reader.stem.json') as f:
        data = json.load(f)
    for item in data:
        data[item] = flatten(data[item])
    return data


def buildDFVector(tfmatrix):
    i = 0
    j = 0
    nl = len(tfmatrix)
    nc = len(tfmatrix[0])
    for i in range(nl):
        for j in range(nc):
            if tfmatrix[i][j] > 1:
                tfmatrix[i][j] = 1
    special_matrix = np.array(tfmatrix)
    return special_matrix.sum(axis=0)


def calcAvgDocSize(docs_list):
    d = {}
    count = 0
    sum = 0
    for doc in docs_list:
        doc_len = len(doc)
        d[count] = doc_len
        sum += doc_len
        count += 1
    return sum / count, count, d


def dummy_func(doc):
    return doc


def buildTFmatrix(docs):
    dict = {}
    id = 0
    cv = CountVectorizer(tokenizer=dummy_func, preprocessor=dummy_func)
    cvfit = cv.fit_transform(docs)
    for candidate in cv.get_feature_names():
        dict[candidate] = id
        id += 1
    return dict, cvfit.toarray()


def allEdges(l):
    comb = combinations(l, 2)
    return list(comb)


def getCandidatesNG(docs_list):
    docs = []

    for doc in docs_list:  # or for sentence in doc, depends on input
        candidateList = []

        doc_tokenize = nltk.word_tokenize(doc)
        candidateList += doc_tokenize

        bigrams = [' '.join(b) for b in nltk.bigrams(doc.split())]
        candidateList += bigrams

        trigrams = [' '.join(b) for b in nltk.trigrams(doc.split())]
        candidateList += trigrams

        docs.append(candidateList)
    return docs


def rankScores(docs):
    b = 0.75
    k1 = 1.2

    finalbm25 = []
    finaltf = []
    finalLen = []
    finalNG = []

    candidates = getCandidatesNG(docs)

    dict, tfMatrix = buildTFmatrix(candidates)
    avgdl, N, dlen = calcAvgDocSize(docs)
    dfVector = buildDFVector(tfMatrix)

    for i in range(len(candidates)):
        scoresbm25 = {}
        scorestf = {}
        scoreslen = {}
        scoresng = {}

        for t in candidates[i]:
            a = (N - dfVector[dict[t]] + 0.5) / (dfVector[dict[t]] + 0.5)

            idfp = math.log(a, 10)  # parte da esquerda

            tf = tfMatrix[i][dict[t]] / len(candidates[i])

            b = (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * (dlen[i] / avgdl))))

            candidateLen = 1
            if use_candidateLen == 1:
                candidateLen = len(t)

            scoresbm25[t] = idfp * b * candidateLen
            scorestf[t] = tf
            scoreslen[t] = len(t)
            scoresng[t] = countGram(t)

        sorted_x_bm25 = sorted(scoresbm25.items(), key=lambda kv: kv[1], reverse=True)
        sorted_dict_bm25 = collections.OrderedDict(sorted_x_bm25)
        new_sorted_dict_bm25 = {}
        for i, c in enumerate(list(sorted_dict_bm25.keys())):
            new_sorted_dict_bm25[c] = i
        finalbm25.append(new_sorted_dict_bm25)

        sorted_x_tf = sorted(scorestf.items(), key=lambda kv: kv[1], reverse=True)
        sorted_dict_tf = collections.OrderedDict(sorted_x_tf)
        new_sorted_dict_tf = {}
        for i, c in enumerate(list(sorted_dict_tf.keys())):
            new_sorted_dict_tf[c] = i
        finaltf.append(new_sorted_dict_tf)

        sorted_x_len = sorted(scoreslen.items(), key=lambda kv: kv[1], reverse=True)
        sorted_dict_len = collections.OrderedDict(sorted_x_len)
        new_sorted_dict_len = {}
        for i, c in enumerate(list(sorted_dict_len.keys())):
            new_sorted_dict_len[c] = i
        finalLen.append(new_sorted_dict_len)

        sorted_x_ng = sorted(scoresng.items(), key=lambda kv: kv[1], reverse=True)
        sorted_dict_ng = collections.OrderedDict(sorted_x_ng)
        new_sorted_dict_ng = {}
        for i, c in enumerate(list(sorted_dict_ng.keys())):
            new_sorted_dict_ng[c] = i
        finalNG.append(new_sorted_dict_ng)

    return finalbm25, finaltf, finalLen, finalNG


def RRFScore(candidate, bm25ranks, tfidf_vector, tfranks, lenranks, ngranks, prranks):
    sum = 0

    if r_usebm25:
        sum += (1/(50+bm25ranks[candidate]))

    if r_usetfidf:
        if candidate not in tfidf_vector:
            posTFIDF = 10000000000
        else:
            for i, c in enumerate(tfidf_vector):
                if candidate == c:
                    posTFIDF = i
        sum += (1/(50+posTFIDF))

    if r_usetf:
        sum += (1 / (50 + tfranks[candidate]))

    if r_uselen:
        sum += (1/(50+lenranks[candidate]))

    if r_useng:
        sum += (1/(50+ngranks[candidate]))

    if r_usepr:
        if candidate not in prranks.keys():
            sum += 1/500000000
        else:
            sum += (1/(50 + prranks[candidate]))

    return sum


def apk(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def createGraph(doc_ngrams):
    prrank = []
    for cdoc in doc_ngrams:

        G = nx.Graph()  # grafo em que cada node eh um candidato e uma edge a ligacao entre eles

        for i in range(0, len(cdoc)):
            lencandidates = len(cdoc[i])
            count = 0
            l = []
            for c in cdoc[i]:
                G.add_node(c)
                count += 1
                l.append(c)
                if count == lencandidates:  # se estiver no ultimo candidato da frase, adiciono as edges
                    G.add_edges_from(allEdges(l))

        pr = nx.pagerank(G, alpha=0.15, max_iter=50)  # alpha eh o d

        sorted_x_pr = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)
        sorted_dict_pr = collections.OrderedDict(sorted_x_pr)
        new_sorted_dict_pr = {}
        for i, c in enumerate(list(sorted_dict_pr.keys())):
            new_sorted_dict_pr[c] = i

        prrank.append(pr)

    return prrank


def main():
    dict_XML, filenames = readXML()
    dict_json = readJson()

    allSentencesPerDoc = []
    for doc in dict_XML.values():
        doclist = [line for line in doc]
        docstr = ''.join(doclist)
        sentences = re.split(r'[.!?]',
                             docstr)  # lista de frases (visto que o factor de similaridade eh estarem na mesma frase
        sentences_XML_preprocessed = preProcessing(sentences)
        sentences_XML_without_sw = removeStopWords(sentences_XML_preprocessed)
        allSentencesPerDoc.append(sentences_XML_without_sw)

    doc_ngrams = []
    for d in allSentencesPerDoc:
        doc_ngrams.append(getCandidatesNG(d))

    prrank = createGraph(doc_ngrams)

    docs_XML_preprocessed = preProcessing(dict_XML.values())
    docs_XML_without_sw = removeStopWords(docs_XML_preprocessed)

    bm25rank, tfrank, lenrank, ngrank = rankScores(docs_XML_without_sw)

    tfidf_XML_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))

    tfidf_XML_vectors = tfidf_XML_vectorizer.fit_transform(docs_XML_without_sw)

    candidatesPerDoc = getCandidatesNG(docs_XML_without_sw)

    avgPrecList = []
    docNum = 0

    train, test = train_test_split(filenames, train_size=0.7, random_state=37)

    for f in filenames:

        finalScoreDoc = []

        tfidf_XML_docNum = tfidf_XML_vectors[docNum]

        tfidf_XML_docNum_table = pd.DataFrame(tfidf_XML_docNum.T.todense(), index=tfidf_XML_vectorizer.get_feature_names(), columns=["tfidf"])
        tfidf_XML_docNum_table = tfidf_XML_docNum_table.sort_values(by=["tfidf"], ascending=False)

        tfidf_vector = list(tfidf_XML_docNum_table.index)

        for c in set(candidatesPerDoc[docNum]):
            finalScoreDoc.append([RRFScore(c, bm25rank[docNum], tfidf_vector, tfrank[docNum],
                                            lenrank[docNum], ngrank[docNum], prrank[docNum]), c])

        finalScoreDoc = sorted(finalScoreDoc, reverse=True)

        bestCandidates = []
        for i in range(10):
            bestCandidates.append(finalScoreDoc[i][1])


        avgPrec = mapk(dict_json[f], bestCandidates, 10)

        print("File Number:", docNum+1)
        print(bestCandidates)
        print("Average Precision:", avgPrec)
        print("\n")

        avgPrecList.append(avgPrec)
        docNum += 1


    map = mean(avgPrecList)
    print("Mean Average Precision:", map)

main()