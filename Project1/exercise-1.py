from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

import re
import pandas as pd

all_docs = fetch_20newsgroups()
docs = all_docs.data


def removeStopWords (list):
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
        l = l.lower()          # put lower key
        l = re.sub("(.)*@(.)*", '', l)  # remove emails
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)]", '', l)   #remove pontuation
        l = re.sub("\d+", '', l)   #remove numbers
        docs.append(str(l))
    return docs


def getLen(candidate):
    lenList =[]
    for i in candidate:
        lenList.append(len(i))
    return lenList

def main():
    f = open('baseball.txt', 'r')
    r = ''
    for i in f:
        r += ''.join(i)
    f.close()

    docs.insert(0, r)   #inserir o nosso doc no inicio

    docs_preprocessed = preProcessing(docs)
    docs_without_sw = removeStopWords(docs_preprocessed)

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))

    tfidf_vectors = tfidf_vectorizer.fit_transform(docs_without_sw)

    first_doc = tfidf_vectors[0]

    first_doc_table = pd.DataFrame(first_doc.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

    lenList = getLen(tfidf_vectorizer.get_feature_names())
    first_doc_table["len"] = lenList
    first_doc_table["score"] = first_doc_table.tfidf * first_doc_table.len
    first_doc_table = first_doc_table.sort_values(by=["score"], ascending=False)

    print(first_doc_table.head(5))


main()
