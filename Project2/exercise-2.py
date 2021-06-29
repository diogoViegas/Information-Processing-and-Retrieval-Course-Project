from xml.dom import minidom

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import CountVectorizer

import re
import os
import json
import math
import operator
import pandas as pd
import networkx as nx
import collections
import numpy as np
from itertools import combinations
from statistics import mean


ps = PorterStemmer()

use_candidateLen = 1    # 0 - nao usa o tamanho do candidato || 1 - usa tamanha do candidato (#caracteres)

weightType = 1          # 0 - sem pesos || 1 - co-occurence || 2 - embeddings
priorsType = 1          # 0 - sem priors || 1 - utiliza len do candidato ||
                        # 2 - utiliza posicao do candidato || 3 - utiliza o BM25 para calcular priors


def wordEmbeddings(candidates):
    model1 = gensim.models.Word2Vec(candidates, min_count=1,
                                    size=50, window=5, sg = 1)

    candidates = list(set(flatten(candidates)))

    mat = np.zeros((len(candidates), len(candidates)))

    for i in range(0,len(mat)):
        for j in range(0, len(mat[i])):
            mat[i][j] = model1.wv.similarity(candidates[i], candidates[j])

    mat = pd.DataFrame(mat)
    mat.index = candidates
    mat.columns = candidates
    return mat


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
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)|\'|—|$|’|“|”|`|\-]", '', l)   #remove pontuation
        l = re.sub("\d+", '', l)  # remove numbers
        docs.append(str(l))
    return docs


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
            s += word.firstChild.data
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
    return sum/count, count, d


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


def weights(ctxs, candidates):
    mat = np.zeros((len(candidates), len(candidates)))

    nei = []
    nei_size = 3

    for ctx in ctxs:
        words = ctx.split(' ')

        for i, _ in enumerate(words):
            nei.append(words[i])

            if len(nei) > (nei_size * 2) + 1:
                nei.pop(0)

            pos = int(len(nei) / 2)
            for j, _ in enumerate(nei):
                if nei[j] in candidates and words[i] in candidates:
                    mat[candidates.index(nei[j]), candidates.index(words[i])] += 1

    mat = pd.DataFrame(mat)
    mat.index = candidates
    mat.columns = candidates
    return mat


def allEdgesWeighted(l, matrix):
    comb = combinations(l, 2)
    combList = list(comb)
    output = []
    for c in combList:

        if matrix[c[0]][c[1]] < 0:
            tupleWeighted = c + (0,)
        else:
            tupleWeighted= c + (matrix[c[0]][c[1]],)

        output.append(tupleWeighted)
    return output


def allEdges(l):
    comb = combinations(l, 2)
    return list(comb)


def getCandidatesNG(docs_list):
    docs = []

    for doc in docs_list:   #or for sentence in doc, depends on input
        candidateList = []

        doc_tokenize = nltk.word_tokenize(doc)
        candidateList += doc_tokenize

        bigrams = [' '.join(b) for b in nltk.bigrams(doc.split())]
        candidateList += bigrams

        trigrams = [' '.join(b) for b in nltk.trigrams(doc.split())]
        candidateList += trigrams

        docs.append(candidateList)
    return docs


def BM25(docs):
    b = 0.75
    k1 = 1.2
    final = []

    candidates = getCandidatesNG(docs)

    dict, tfMatrix = buildTFmatrix(candidates)
    avgdl, N, dlen = calcAvgDocSize(docs)
    dfVector = buildDFVector(tfMatrix)

    for i in range(len(candidates)):
        scores = {}
        for t in candidates[i]:
            a = (N - dfVector[dict[t]] + 0.5) / (dfVector[dict[t]] + 0.5)

            idfp = math.log(a, 10)  #parte da esquerda

            tf = tfMatrix[i][dict[t]] / len(candidates[i])

            b = (tf * (k1+1)) / (tf + k1 * (1-b + (b*(dlen[i]/avgdl) ) )  )

            candidateLen = 1
            if use_candidateLen == 1:
                candidateLen = len(t)

            scores[t] = idfp * b * candidateLen
        score = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        sorted_dic = collections.OrderedDict(score)
        final.append(sorted_dic)

    return final


def priorsLenOrPos(doc_ngrams):
    listCan = []

    for docC in doc_ngrams:
        dictCan = {}
        sentNum = 1
        for sentenceC in docC:
            for c in sentenceC:
                if priorsType == 2:
                    if c not in dictCan.keys():
                        dictCan[c] = 1/sentNum
                elif priorsType == 1:
                    dictCan[c] = len(c)
            sentNum += 1
        listCan.append(dictCan)

    return listCan


def main():
    dict_XML, filenames = readXML()
    dict_json = readJson()

    allSentencesPerDoc = []
    for doc in dict_XML.values():
        doclist = [line for line in doc]
        docstr = ''.join(doclist)
        sentences = re.split(r'[.!?]', docstr)  #lista de frases (visto que o factor de similaridade eh estarem na mesma frase
        sentences_XML_preprocessed = preProcessing(sentences)
        sentences_XML_without_sw = removeStopWords(sentences_XML_preprocessed)
        allSentencesPerDoc.append(sentences_XML_without_sw)

    doc_ngrams = []
    matrixList = []

    for d in allSentencesPerDoc:
        doc_ngrams.append(getCandidatesNG(d))

    if weightType == 2:
        for candidates in doc_ngrams:
            matrixList.append(wordEmbeddings(candidates))

    docs_XML_preprocessed = preProcessing(dict_XML.values())
    docs_XML_without_sw = removeStopWords(docs_XML_preprocessed)

    if priorsType == 3:
        priors = BM25(docs_XML_without_sw)
    elif priorsType == 1 or priorsType == 2:
        priors = priorsLenOrPos(doc_ngrams)

    docNum = 0
    avgPrecList = []
    for cdoc in doc_ngrams:
        candidates = list(set(flatten(cdoc)))

        matrix = weights(allSentencesPerDoc[docNum], candidates)

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
                    if weightType == 0:
                        G.add_edges_from(allEdges(l))
                    elif weightType == 1:
                        G.add_weighted_edges_from(allEdgesWeighted(l, matrix))
                    else:
                        G.add_weighted_edges_from(allEdgesWeighted(l, matrixList[docNum]))

        if priorsType == 0:
            pr = nx.pagerank(G, alpha=0.15, max_iter=50)  # alpha eh o d
        else:
            pr = nx.pagerank(G, alpha=0.15, personalization=priors[docNum], max_iter= 50)  # alpha eh o d

        sorted_x = sorted(pr.items(), key=lambda kv: kv[1])  # sort aos values do dicionario
        sorted_dic = collections.OrderedDict(sorted_x)

        bestCandidates = []
        for i in range(10):
            pair = sorted_dic.popitem()
            bestCandidates.append(pair[0])

        avgPrec = mapk(dict_json[filenames[docNum]], bestCandidates, 10)

        print("File Number:", docNum+1)
        print(bestCandidates)
        print("Average Precision:", avgPrec)
        print("\n")
        avgPrecList.append(avgPrec)
        docNum += 1

    map = mean(avgPrecList)
    print("Mean Average Precision:", map)


main()
