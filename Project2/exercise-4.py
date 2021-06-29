import collections
from xml.dom import minidom
import xml.etree.ElementTree as et

import networkx as nx
import nltk
from nltk.corpus import stopwords
from itertools import combinations
import re

def readXML():
    mydoc = minidom.parse('Europe.xml')

    Title_Desc = []

    tree = et.parse('Europe.xml')

    root = tree.getroot()

    for item in root.findall('./channel/item'):

        Title_Desc.append([item.find('title').text, item.find('description').text])

    return Title_Desc


def flatten(list):
   flat_list = []
   for subslist in list:
       for item in subslist:
           flat_list.append(item)
   return flat_list


def removeStopWords(list):
    docs = []
    stopWordList = set(stopwords.words("english"))
    stopWordList = stopWordList.union(['\'s', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'briefing'])
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
        l = re.sub("[?|\.|!|:|,|;|<|>|%|\[|\]|(|)|\'|—|$|’|“|”]", '', l)   #remove pontuation
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


def allEdges(l):
    comb = combinations(l, 2)
    return list(comb)


def creatWordCloud(wordsList):
    d = open("wordCloud.html", "w")
    message = """<html>
    <meta charset="utf-8">
    
    <!-- Load d3.js -->
    <script src="https://d3js.org/d3.v4.js"></script>
    
    <!-- Load d3-cloud -->
    <script src="https://cdn.jsdelivr.net/gh/holtzy/D3-graph-gallery@master/LIB/d3.layout.cloud.js"></script>
    
    <!-- Create a div where the graph will take place -->
    <div id="my_dataviz" style="left: 300px; top: 20px; position: absolute;"></div>
    
    <script>
    // List of words
    
     
    var myWords = [{word: "%s", size: "70"}, {word: "%s", size: "67"}, {word: "%s", size: "64"}, {word: "%s", size: "61"},
                    {word: "%s", size: "58"}, {word: "%s", size: "55"}, {word: "%s", size: "52"}, {word: "%s", size: "49"},
                    {word: "%s", size: "46"}, {word: "%s", size: "43"}, {word: "%s", size: "40"}, {word: "%s", size: "37"},
                    {word: "%s", size: "34"}, {word: "%s", size: "31"}, {word: "%s", size: "28"}, {word: "%s", size: "25"},
                    {word: "%s", size: "22"}, {word: "%s", size: "19"}, {word: "%s", size: "16"}, {word: "%s", size: "13"}]
    var titles = %s
    console.log(titles)
    colors = ["#11144C", "#3A9679", "#FABC60", "#E16262"]
    
    // set the dimensions and margins of the graph
    var margin = {top: 10, right: 10, bottom: 10, left: 10},
        width = 1200 - margin.left - margin.right,
        height = 950 - margin.top - margin.bottom;
    
    // append the svg object to the body of the page
    var svg = d3.select("#my_dataviz").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");
    
    // Constructs a new cloud layout instance. It run an algorithm to find the position of words that suits your requirements
    // Wordcloud features that are different from one word to the other must be here
    var layout = d3.layout.cloud()
      .size([width, height])
      .words(myWords.map(function(d) { return {text: d.word, size:d.size}; }))
      .padding(5)        //space between words
      .rotate(function() { return ~~(Math.random() * 2) * 90; })
      .fontSize(function(d) { return d.size; })      // font size of words
      .on("end", draw);
    layout.start();
    
    // This function takes the output of 'layout' above and draw the words
    // Wordcloud features that are THE SAME from one word to the other can be here
    function draw(words) {
      svg
        .append("g")
          .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
          .selectAll("text")
            .data(words)
          .enter().append("text")
            .style("font-size", function(d) { return d.size; })
            .style("fill", function() { return colors[Math.floor(Math.random()*colors.length)] })
            .attr("text-anchor", "middle")
            .style("font-family", "Impact")
            .attr("transform", function(d) {
              return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
            })
            .text(function(d) { return d.text; })
            .append("title").text(function(d){return 'News Titles:\\n'  + titles[d.text]});
    }
    </script>
    
    </html>"""

    messagefinal = message % wordsList

    d.write(messagefinal)
    d.close()

def readTitles():
    mydoc = minidom.parse('Europe.xml')

    Title_Desc = []

    tree = et.parse('Europe.xml')

    root = tree.getroot()

    for item in root.findall('./channel/item'):
        Title_Desc.append([item.find('title').text])

    return Title_Desc


def getTitles(news):

    allTitles = readTitles()

    returnTitles = []
    for titles in news:
        auxTitles = []
        for title in titles:
            auxTitles.append(allTitles[title])
        returnTitles.append(auxTitles)

    return returnTitles


def main():

    title_desc = readXML()
    candidates = []

    for tD in title_desc:

        title_preprocessed = preProcessing([tD[0]])
        title_without_sw = removeStopWords(title_preprocessed)
        title_candidates = getCandidatesNG(title_without_sw)
        desc_preprocessed = preProcessing([tD[1]])

        desc_without_sw = removeStopWords(desc_preprocessed)
        desc_candidates = getCandidatesNG(desc_without_sw)

        td_candidates = flatten(title_candidates) + flatten(desc_candidates)

        candidates.append(td_candidates)

    G = nx.Graph()  # grafo em que cada node eh um candidato e uma edge a ligacao entre eles
    for i in range(0, len(candidates)):
        lencandidates = len(candidates[i])
        count = 0
        l = []
        for c in candidates[i]:
            G.add_node(c)
            count += 1
            l.append(c)
            if count == lencandidates:  # se estiver no ultimo candidato da frase, adiciono as edges
                G.add_edges_from(allEdges(l))  # combinacoes

    pr = nx.pagerank(G, 0.15)  # 0.15 eh o d
    sorted_x = sorted(pr.items(), key=lambda kv: kv[1])  # sort aos values do dicionario
    sorted_dic = collections.OrderedDict(sorted_x)

    listSorted = list(sorted_dic)

    wordsList = ()
    for i in range(20):  # 20 melhores
        wordsList += (listSorted[i],)
    keyWordsNews = []
    countList = []
    for word in wordsList:
        whereAmI = []
        count=0
        for i in range(len(title_desc)):
            wordsOfKeyword = word.split()

            for wrd in wordsOfKeyword:

                if wrd.capitalize() in title_desc[i][0] or wrd.capitalize() in title_desc[i][1] or wrd in title_desc[i][0] or wrd in title_desc[i][1]:
                    whereAmI += [i]
                    count+=1
        countList.append(count)
        keyWordsNews.append(list(set(whereAmI)))

    titles = getTitles(keyWordsNews)
    newtitles = {}
    for i,title in enumerate(titles):
        titleString = ''
        for s in flatten(title):

            titleString += s + '\n'

        newtitles[wordsList[i]] = titleString + '\n Number of Appearances: \n' + str(countList[i])

    wordsList = wordsList + (newtitles,)
    creatWordCloud(wordsList)

main()