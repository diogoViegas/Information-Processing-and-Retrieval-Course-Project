from xml.dom import minidom

from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall

import re
import os
import numpy as np
import math
import operator
import json

ps = PorterStemmer()


# Flags que alternam a abordagem

#Caso seja utilizado o Reconhecimento de Entidade Nominal(candidateType = 1) a flags is_steam e is_lower devem ser a 0

is_stem = 0          # 0 - nao aplica stem || 1 - aplica stem
is_lower = 0         # 0 - nao aplica lowercase || 1 - aplica lowercase
use_candidateLen = 0 # 0 - nao usa o tamanho do candidato || 1 - usa tamanha do candidato (#caracteres)
candidateType = 1    # 0 - Expressao Regular || 1 - Reconhecimento de Entidade nominal || 2 - n-grams [1,3]


def fmeasure(precision,recall):
    if precision != 0 or recall != 0:
        return (2*precision*recall) / (precision+recall)
    return 0


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
        if is_lower == 1:
            l = l.lower()          # put lower key
        l = re.sub("(.)*@(.)*", '', l)  # remove emails
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)]", '', l)   #remove pontuation
        l = re.sub("\d+", '', l)   #remove numbers
        docs.append(str(l))
    return docs


def readXML():
    docs = {}
    i = 0
    docname = []
    for filename in os.listdir("train"):
        i += 1
        s = ''
        mydoc = minidom.parse("train/"+filename)
        words = mydoc.getElementsByTagName('word')
        filename_X = ''
        for c in filename:
            if c == '.':
                break
            filename_X += str(c)
        docname.append(filename_X)

        for word in words:
            if is_stem == 1:
                s += ps.stem(word.firstChild.data) #stemming
            else:
                s += word.firstChild.data
            s += ' '

        docs[filename_X] = s

    #docs eh uma lista de strings em que cada string e um doc
    return docs, docname


def getCandidatesNG(docs_list):
    docs = []

    for doc in docs_list:
        candidateList = []
        doc_tokenize = nltk.word_tokenize(doc)
        candidateList += doc_tokenize

        bigrams = [' '.join(b) for b in nltk.bigrams(doc.split())]
        candidateList += bigrams

        trigrams = [' '.join(b) for b in nltk.trigrams(doc.split())]
        candidateList += trigrams

        docs.append(candidateList)
    return docs


def getCandidatesRE(docs_list):
    docs = []

    chunker = nltk.RegexpParser(r'''
    C:
    {((<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+)+}
    ''')

    for doc in docs_list:
        doc_tokenize = nltk.word_tokenize(doc)
        doc_tags = nltk.pos_tag(doc_tokenize)
        c = chunker.parse(doc_tags)
        Cs = list(c.subtrees(filter=lambda x: x.label()=='C'))

        candidate_list = []

        for branch in Cs:
            s = ''
            for word in branch.leaves():
                s += word[0] + ' '

            candidate_list.append(s[:-1])

        docs.append(candidate_list)

    return docs


def getCandidatesNE(docs_list):
    docs = []

    for doc in docs_list:
        doc_tokenize = nltk.word_tokenize(doc)
        doc_tags = nltk.pos_tag(doc_tokenize)
        c = nltk.ne_chunk(doc_tags, binary=True)
        Cs = list(c.subtrees(filter=lambda x: x.label()=='NE'))

        candidate_list = []

        for branch in Cs:
            s = ''
            for word in branch.leaves():
                s += word[0] + ' '

            candidate_list.append(s[:-1].lower())

        docs.append(candidate_list)

    return docs


def BM25(docs):
    b = 0.75
    k1 = 1.2
    final = []

    if candidateType == 0:
        candidates = getCandidatesRE(docs)
    elif candidateType == 1:
        candidates = getCandidatesNE(docs)
    else:
        candidates = getCandidatesNG(docs)

    dict, tfMatrix = buildTFmatrix(candidates)
    avgdl , N , dlen = calcAvgDocSize(docs)
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
        final.append(score)
    return final


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


def readJson():
    if is_stem == 0:
        filename = 'test.reader.json'
    else:
        filename = 'test.reader.stem.json'
    with open (filename) as f:
        data = json.load(f)
    for item in data:
        data[item] = flatten(data[item])
    return data


def main():
    precision5List = []
    recallList = []
    f1List = []
    finalList = []

    dict_XML, filenames = readXML()
    dict_json = readJson()

    docs_XML_preprocessed = preProcessing(dict_XML.values())
    docs_XML_without_sw = removeStopWords(docs_XML_preprocessed)

    bm25 = BM25(docs_XML_without_sw)

    for f in bm25:
        j = 0
        final = []
        for l in f:
            if j < 5:
                final.append(l[0])
            j += 1
        finalList.append(final)

    i = 0

    for f in dict_XML.keys():

        values_5 = finalList[i]    # 5 primeiros candidates

        precision5 = precision(set(values_5), set(dict_json[f]))
        precision5List.append(precision5)

        recall1 = recall(set(values_5), set(dict_json[f]))
        recallList.append(recall1)

        f1 = fmeasure(precision5, recall1)
        f1List.append(f1)

        i += 1

    len_files = len(filenames)
    AP5 = sum(precision5List) / len_files
    ARecall = sum(recallList) / len_files
    AF1 = sum(f1List) / len_files
    print("The Average Recall is " + str( round(ARecall,4)) + " and the Average F1 is " + str(round(AF1,4)) + " and the Average Precision@5 is " + str(round(AP5,4)) + "\n")


main()
