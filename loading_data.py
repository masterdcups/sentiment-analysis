#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Loading xml dataset --- OK
# PS : Go to affectivetext_test.xml and replace & by &amp;

from xml.dom import minidom
import pandas as pd

def _SemEval2007_Path2String(X_path, y_path):
    #print('hello before minidom parse')
    xmldoc = minidom.parse(X_path)
    itemlist = xmldoc.getElementsByTagName('instance')

    file = open(y_path)
    lines = file.read().split('\n')
    #print(len(lines))
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
    DATA_X_PATH1 = "../data/AffectiveText.test/affectivetext_test.xml"
    DATA_Y_PATH1 = "../data/AffectiveText.test/affectivetext_test.emotions.gold"

    DATA_X_PATH2 = "../data/AffectiveText.trial/affectivetext_trial.xml"
    DATA_Y_PATH2 = "../data/AffectiveText.trial/affectivetext_trial.emotions.gold"

    X1, y1 = _SemEval2007_Path2String(DATA_X_PATH1, DATA_Y_PATH1)
    X2, y2 = _SemEval2007_Path2String(DATA_X_PATH2, DATA_Y_PATH2)
    # y.update(y2)
    # print(X1)
    # print(y1)

    df = pd.DataFrame([X1, y1])

    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    for column in df:  # retenir le score maximal
        df[column][1] = emotions[df[column][1].index(max(df[column][1]))]
        # print(df[column])

    return df.T


# In[9]:


# Loading arff dataset
import arff
import pandas as pd

def loadDEFT2015():
    df = pd.DataFrame(columns=['text', 'emotion'])
    data = list(arff.load('Corpus.de.tweets.en.français/DEFT2015-arff/Train2.2.arff'))

    for i in range(len(data)):
        item = data[i]
        df.loc[i] = [item[0], item[1]]

    return df


# In[14]:


# loading txt dataset --- OK 

def loadEmotionTweet(train_path, dev_path):
    train = pd.read_csv(train_path,sep='\t',index_col=[0])
    dev = pd.read_csv(dev_path,sep='\t',index_col=[0])
    train = pd.concat([train,dev])
    train = train.sample(frac=1)  # just shuffle
    #train['Tweet'] = train['Tweet'].apply(lambda x:x.lower())

    return train


# In[15]:


df = loadEmotionTweet("../data/EmotionTweet/2018-E-c-En-train.txt", "../data/EmotionTweet/2018-E-c-En-dev.txt")
df.to_pickle('data/final/NotIot_EN_tweets.p') 


# In[6]:


df2 = Load_SemEval2007Data()
df2.to_pickle('data/final/NotIot_EN_News.p')


# In[5]:


import pandas as pd
dff = pd.read_pickle("loadingScripts/loaded_datasets/Test2.2.pkl")
dff.to_pickle('data/final/NotIot_FR_tweets_test.p')


# In[46]:


csv_file = "data/french_iot_tweets.csv"
df = pd.read_csv(csv_file, header=None)
df = df.iloc[:,[0,6,7]]
df.rename(index=str, columns={6: "a", 7: "c"})
df


# In[47]:


df.to_pickle('data/final/Iot_FR_tweets.p')


# In[52]:


csv_file = "../code_news/news_en.csv"
df = pd.read_csv(csv_file, header=None, encoding='utf-8')
df


# In[ ]:




