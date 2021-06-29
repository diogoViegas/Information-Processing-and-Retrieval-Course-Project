from xml.dom import minidom

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import re
import os
import pandas as pd
import numpy as np
import json
from statistics import mean
import math

ps = PorterStemmer()

# Flags que alternam a abordagem

#Caso seja utilizado o Reconhecimento de Entidade Nominal(candidateType = 1) a flags is_steam e is_lower devem ser a 0

is_stem = 0         # 0 - nao aplica stem || 1 - aplica stem
is_lower = 0        # 0 - nao aplica lowercase || 1 - aplica lowercase
use_countNgram = 1  # 0 - nao usa o numero de palavras como feature || 1 - usa o numero de palavras como feature
candidateType = 2   # 0 - Expressao Regular || 1 - Reconhecimento de Entidade Nominal || 2 - n-grams [1,3]

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


def readXML():
    docs = {}
    i = 0
    docsname = []
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
        docsname.append(filename_X)

        for word in words:
            if is_stem == 1:
                s += ps.stem(word.firstChild.data)  # stemming
            else:
                s += word.firstChild.data
            s += ' '

        docs[filename_X] = s

    #docs eh uma lista de strings em que cada string e um doc
    return docs, docsname

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
            l = l.lower()  # put lower key
        l = re.sub("(.)*@(.)*", '', l)  # remove emails
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)|_]", '', l)   #remove pontuation
        l = re.sub("\d+", '', l)   #remove numbers
        docs.append(str(l))
    return docs

def buildFeaturesTable(docs, filenames):
    table = []

    if candidateType == 2:
        candidates = getCandidatesNG(docs)
    elif candidateType == 1:
        candidates = getCandidatesNE(docs)
    else:
        candidates = getCandidatesRE(docs)
    dict_json = readJson()

    dict, tfMatrix = buildTFmatrix(candidates)

    dfVector = buildDFVector(tfMatrix)

    for i in range(len(candidates)):
        pos = 0
        for term in candidates[i]:
            is_key_phrase = 0
            pos += 1
            table_row = []
            #posCand = int(pos/len(candidates[i])*3)    #posicao do candidato na lista de candidatos
            tf = candidates[i].count(term) / len(candidates[i])
            if term in dict_json[filenames[i]]:
                is_key_phrase = 1
            if use_countNgram == 1:
                table_row += [term, len(term), countGram(term),  tf, is_key_phrase]
            else:
                table_row += [term, len(term),  tf, is_key_phrase]

            table.append(tuple(table_row))
    return table


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


def overSampling(unbal, col_name):

    target_count = unbal[col_name].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    RANDOM_STATE = 44
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1 - ind_min_class]]}

    df_class_min = unbal[unbal[col_name] == min_class]
    df_class_max = unbal[unbal[col_name] != min_class]


    df_over = df_class_min.sample(len(df_class_max), replace=True, random_state=RANDOM_STATE)
    balanced = df_class_max.append(df_over, ignore_index=True)
    values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]

    return balanced


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


def main():
        dict_XML, filenames = readXML()
        dict_json = readJson()
        apList = []

        trainf, testf = train_test_split(filenames, train_size=0.7, random_state=37)

        dict_Train = { key: dict_XML[key] for key in trainf}

        classificationTableTrain = buildFeaturesTable(dict_Train.values(), trainf)
        dfTrain = pd.DataFrame(classificationTableTrain, columns=['term', 'n', 'ngrams', 'tf', 'keyPhrase'])

        dfTrain.pop('term').values

        train_balanced = overSampling(dfTrain, 'keyPhrase')

        y_train: np.ndarray = train_balanced.pop('keyPhrase').values
        X_train: np.ndarray = train_balanced.values

        # Criacao do perceptrao
        ppn = Perceptron(random_state=37)

        # Treino do perceptrao
        ppn.fit(X_train, y_train)

        for f in testf:
            print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
            print(dict_XML[f])
            classificationTableTest = buildFeaturesTable([dict_XML[f]], [f])
            dfTest = pd.DataFrame(classificationTableTest, columns=['term', 'n', 'ngrams', 'tf', 'keyPhrase'])

            test_term = dfTest.pop('term').values
            y_test: np.ndarray = dfTest.pop('keyPhrase').values
            X_test: np.ndarray = dfTest.values

            # Aplicar o treino ao perceptrao com o conjunto X e fazer as previsoes para o conjunto y
            y_pred = ppn.predict(X_test)
            y_predProb = ppn._predict_proba_lr(X_test)
            #print(y_predProb)
            probClass1 = y_predProb[:, 1]



            term_prob = list(zip(test_term, probClass1))
            sorted_term_prob = sorted(term_prob, key=lambda tup: tup[1], reverse=True)
            print("Highest scoring candidates:")
            print(sorted_term_prob[:10])
            print('##########################################################################################')

            print('Perceptron Accuracy: %.3f' % accuracy_score(y_pred, y_test))

            bestCandidates = []
            for i in range(10):
                bestCandidates.append(sorted_term_prob[i][0])

            ap = mapk(dict_json[f], bestCandidates, 10)
            apList.append(ap)

        print(mean(apList))


main()