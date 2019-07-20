#Feature engine
import random

random.seed = 16
import pandas as pd
from gensim.models.word2vec import Word2Vec

data = pd.read_csv("./input/sentiment_analysis_trainingset.csv")

stopwords = []
with open("../my_solution/input/stop_words.txt", encoding='gbk') as f:
    for line in f.readlines():
        line = line.strip()
        stopwords.append(line)


def segWord(doc):
    seg_list = jieba.cut(doc, cut_all=False)
    return list(seg_list)


# move stop words
def filter_map(arr):
    res = ""
    for c in arr:
        if c not in stopwords and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res += c
    return res


# move stop words and generate char sent
def filter_char_map(arr):
    res = []
    for c in arr:
        if c not in stopwords and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res.append(c)
    return " ".join(res)


# get char of sentence
def get_char(arr):
    res = []
    for c in arr:
        res.append(c)
    return list(res)

def enconding_labels(datasets):
    cut_regions = [-3, -2, -1, 0, 1]
    labels = datasets.iloc[:, 2:]
    labels_cut_regions = {}

    for label in labels:
        labels_cut_regions[label] = cut_regions

    cut_df = pd.DataFrame()
    for field in labels_cut_regions.keys():
        cut_series = pd.cut(datasets[field], labels_cut_regions[field], right=True)
        onehot_df = pd.get_dummies(cut_series, prefix=field)
        cut_df = pd.concat([cut_df, onehot_df], axis=1)
    new_df = pd.concat([datasets, cut_df], axis=1)
    return new_df



data.content = data.content.map(lambda x: filter_map(x))
data.content = data.content.map(lambda x: get_char(x))
data = enconding_labels(data)
data.to_csv("preprocess/train_char.csv", index=None)

validation = pd.read_csv("./input/sentiment_analysis_validationset.csv")
validation.content = validation.content.map(lambda x: filter_map(x))
validation.content = validation.content.map(lambda x: get_char(x))
validation = enconding_labels(validation)
validation.to_csv("preprocess/validation_char.csv", index=None)

test = pd.read_csv("./input/sentiment_analysis_testa.csv")
test.content = test.content.map(lambda x: filter_map(x))
test.content = test.content.map(lambda x: get_char(x))
test = enconding_labels(test)
test.to_csv("preprocess/test_char.csv", index=None)