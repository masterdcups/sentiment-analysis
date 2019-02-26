import pandas as pd


# Loading csv datasets
def load_csv():
    dataset_df = pd.read_csv("text_emotion.csv")
    return dataset_df


# Loading xml dataset
from xml.dom import minidom
import nltk
import collections



def _preprocessing(X):
    max_sentence_len = 0
    word_freqs = collections.Counter()
    for key, val in X.items():
        words = nltk.word_tokenize(val.lower())

        if len(words) > max_sentence_len:
            max_sentence_len = len(words)

        for word in words:
            word_freqs[word] += 1
    return word_freqs, max_sentence_len


def _words2indices_for_a_sentence(X, word2index):
    Xout = {}

    for key, val in X.items():
        words = nltk.word_tokenize(val.lower())
        seqs = []

        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["<UNK>"])

        Xout[key] = seqs

    return Xout


def _SemEval2007_Path2String(X_path, y_path):
    xmldoc = minidom.parse(X_path)
    itemlist = xmldoc.getElementsByTagName('instance')

    file = open(y_path)
    lines = file.read().split('\n')
    lines = lines[0:-1]

    assert len(itemlist) == len(lines), 'data size not uniform'

    X = {}
    y = {}

    for item in itemlist:
        id = int(item.attributes['id'].value)
        X[id] = item.firstChild.nodeValue

    for line in lines:
        items = line.split(' ')
        id = int(items[0])
        y[id] = []
        for i in range(1, len(items)):
            y[id].append(float(items[i]))

    return X, y


def Load_SemEval2007Data():
    DATA_X_PATH1 = "AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml"
    DATA_Y_PATH1 = "AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold"

    DATA_X_PATH2 = "AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml"
    DATA_Y_PATH2 = "AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold"

    X1, y1 = _SemEval2007_Path2String(DATA_X_PATH1, DATA_Y_PATH1)
    X2, y2 = _SemEval2007_Path2String(DATA_X_PATH2, DATA_Y_PATH2)
    # y.update(y2)
    print(X1)
    print(y1)

    df= pd.DataFrame([X1, y1])

    for column in df:         # retenir le score maximal
        df[column][1]= max(df[column][1])
        print(df[column])
        
    return df



# Loading arff dataset

import arff
import pandas as pd

def loadArff:
df = pd.DataFrame(columns=['text', 'emotion'])
data = list(arff.load('Corpus.de.tweets.en.français/DEFT2015-arff/Train2.2.arff'))

for i in range(len(data)):
    item = data[i]
    df.loc[i] = [item[0], item[1]]

return df



# loading txt dataset

import os
from golem.transformers import basic_transformer
from golem.transformers import tweet_contains_tag_transformer
from golem.parsers import csvtojson
from golem.pipeline import pipeline

def loadEmotionTweet():
    parser = csvtojson.CsvToJsonParser('data/data_csv/2018-E-c-En-train.txt')

    if not os.path.exists('data/data_json'):
        os.makedirs('data/data_json')

    parser.export_json('data/data_json/data.json')
    data = parser.get_pandas_data_frame()
    return data
