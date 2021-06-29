import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re
import networkx as nx
import collections
from itertools import combinations

ps = PorterStemmer()

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
        l = l.lower()          # put lower key
        l = re.sub("(.)*@(.)*", '', l)  # remove emails
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)]", '', l)   #remove pontuation
        l = re.sub("\d+", '', l)   #remove numbers
        docs.append(str(l))
    return docs


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


def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def allEdges(l):
    comb = combinations(l, 2)
    return list(comb)


def main():

    file = open("baseball.txt", "r")
    doclist = [line for line in file]
    docstr = ''.join(doclist)
    sentencesStemed = []
    sentences = re.split(r'[.!?]', docstr)  #lista de frases (visto que o factor de similaridade eh estarem na mesma frase

    for f in sentences:
        sentencesStemed.append(stemSentence(f))

    docs_preprocessed = preProcessing(sentencesStemed)
    docs_without_sw = removeStopWords(docs_preprocessed)

    candidates = getCandidatesNG(docs_without_sw)

    G = nx.Graph()   #grafo em que cada node eh um candidato e uma edge a ligacao entre eles
    for i in range(0, len(candidates)):
        lencandidates = len(candidates[i])
        count = 0
        l = []
        for c in candidates[i]:
            G.add_node(c)
            count += 1
            l.append(c)
            if count == lencandidates:   #se estiver no ultimo candidato da frase, adiciono as edges
                G.add_edges_from(allEdges(l))   #combinacoes

    pr = nx.pagerank(G, alpha=0.15, max_iter=50)    #alpha eh o d
    sorted_x = sorted(pr.items(), key=lambda kv: kv[1])     #sort aos values do dicionario
    sorted_dic = collections.OrderedDict(sorted_x)

    for i in range(5):   #5 melhores
        print(sorted_dic.popitem())


main()
