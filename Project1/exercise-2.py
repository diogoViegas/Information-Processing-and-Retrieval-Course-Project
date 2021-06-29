from xml.dom import minidom

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall

import re
import os
import pandas as pd
import json
import numpy as np

ps = PorterStemmer()


def fmeasure(precision, recall):
    if precision != 0 or recall != 0:
        return (2 * precision * recall) / (precision + recall)
    return 0


def apk(actual, predicted, k=10):
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


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


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
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)|']", '', l)  # remove pontuation
        l = re.sub("\d+", '', l)  # remove numbers
        docs.append(str(l))
    return docs


def getLen(candidate):
    lenList = []
    for i in candidate:
        lenList.append(len(i))
    return lenList


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

    # docs eh uma lista de strings em que cada string e um doc
    return docs, docname


def readJson():
    with open('test.reader.stem.json') as f:
        data = json.load(f)
    for item in data:
        data[item] = flatten(data[item])
    return data


def main():
    i = 0
    precision5List = []
    precision10List = []
    recallList = []
    f1List = []
    mapList = []

    dict_XML, filenames = readXML()
    dict_json = readJson()

    docs_XML_preprocessed = preProcessing(dict_XML.values())
    docs_XML_without_sw = removeStopWords(docs_XML_preprocessed)

    tfidf_XML_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 3))

    tfidf_XML_vectors = tfidf_XML_vectorizer.fit_transform(docs_XML_without_sw)

    for f in filenames:
        i_XML_doc = tfidf_XML_vectors[i]

        i_XML_doc_table = pd.DataFrame(i_XML_doc.T.todense(), index=tfidf_XML_vectorizer.get_feature_names(),
                                       columns=["tfidf"])
        lenList = getLen(tfidf_XML_vectorizer.get_feature_names())
        i_XML_doc_table["len"] = lenList
        i_XML_doc_table["score"] = i_XML_doc_table.tfidf * i_XML_doc_table.len
        i_XML_doc_table = i_XML_doc_table.sort_values(by=["score"], ascending=False)

        table_5 = i_XML_doc_table.head(5)
        table_10 = i_XML_doc_table.head(10)

        values_5 = table_5.index.values  # 5 primeiros candidates
        values_10 = table_10.index.values  # 10 primeiros candidates
        map = mapk(dict_json[f], values_10, 10)
        mapList.append(map)
        precision5 = precision(set(values_5), set(dict_json[f]))
        precision5List.append(precision5)
        precision10 = precision(set(values_10), set(dict_json[f]))
        precision10List.append(precision10)

        recall1 = recall(set(values_10), set(dict_json[f]))
        recallList.append(recall1)

        f1 = fmeasure(precision10, recall1)
        f1List.append(f1)

        i += 1

        print("filename: " + str(f) + " with precision@10: " + str(round(precision10, 2)) + " and recall: " + str(
            round(recall1, 2)) + " and f1: " + str(round(f1, 2)) + " and precision@5 " + str(
            round(precision5, 2)) + "and Mean Average Precision " + str(round(map, 2)) + "\n")

    len_files = len(filenames)
    AP5 = sum(precision5List) / len_files
    AP10 = sum(precision10List) / len_files
    ARecall = sum(recallList) / len_files
    AF1 = sum(f1List) / len_files
    MMAP = sum(mapList) / len_files
    print("The Average Precison@10 is " + str(round(AP10, 3)) + " and the Average Recall is " + str(
        round(ARecall, 3)) + " and the Average F1 is " + str(round(AF1, 3)) + " and the Average Precision@5 is " + str(
        round(AP5, 3)) + " and Average Mean Average Precision " + str(round(MMAP, 3)) + "\n")

main()
