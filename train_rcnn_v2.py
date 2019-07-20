#train model using rcnn.  add fast-text pre-trained chinses corpous
from keras.backend.tensorflow_backend import set_session

#Set the GPU sources.
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import random
random.seed = 42
import pandas as pd
from tensorflow import set_random_seed
set_random_seed(42)
from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from keras.layers import *
from classifier_rcnn import TextClassifier
from gensim.models.keyedvectors import KeyedVectors    #load word2vec model.
import pickle
import gc

# import os
# os.environ['OMP_NUM_THREADS'] = '8'  #set the number of threads


def getClassification(arr):
    arr = list(arr)
    res_values = []
    for i in range(20):                                            #the 20 classes.
        if arr[4 * i: 4 * (i+1)].index(max(arr[4 * i: 4 * (i+1)])) == 0:
            res_values.append(-2)
        elif arr[4 * i: 4 * (i+1)].index(max(arr[4 * i: 4 * (i+1)])) == 1:
            res_values.append(-1)
        elif arr[4 * i: 4 * (i+1)].index(max(arr[4 * i: 4 * (i+1)])) == 2:
            res_values.append(0)
        else:
            res_values.append(1)

    return res_values

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = getClassification(self.model.predict(self.validation_data[0]))
        val_targ = getClassification(self.validation_data[1])
        #macro_F1: https://blog.csdn.net/sinat_28576553/article/details/80258619.   一般对于多分类任务来讲，macro 要优于 micro.
        _val_f1 = f1_score(val_targ, val_predict, average="macro")
        _val_recall = recall_score(val_targ, val_predict, average="macro")
        _val_precision = precision_score(val_targ, val_predict, average="macro")
        _val_accuracy = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_accuracy.append(_val_accuracy)
        print(_val_f1, _val_precision, _val_recall, _val_accuracy)
        print("max f1: {}".format(max(self.val_f1s)))
        print("max acc: {}".format(max(self.val_accuracy)))
        return

import pandas as pd

data = pd.read_csv("preprocess/train_char.csv")
#eval() 函数用来执行一个字符串表达式，并返回表达式的值。 eg: eval('2+2') -> 4
data["content"] = [eval(i) for i in data.content]



validation = pd.read_csv("preprocess/validation_char.csv")
validation["content"] = [eval(i) for i in validation.content]

test = pd.read_csv('preprocess/test_char.csv')
test['content'] = [eval(i) for i in test.content]


EMBEDDING_FILE = '/media/chenshixin/My Passport/datasets/wiki.zh.vec'

model_dir = "train_rcnn/"
maxlen = 1000   #the original num is 1000. the running speed is too slow, so using 500
max_features = 20000
batch_size = 128     #the origial num is 128. but it turns out OOM(out of memory), so changed a samll num.
epochs = 20
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(data.content.values) + list(validation.content.values) + list(test.content.values))   #Tokenizer().fit_on_texts(param:list)
X_train = tokenizer.texts_to_sequences(data.content.values)
X_valid = tokenizer.texts_to_sequences(validation.content.values)

input_train =  sequence.pad_sequences(X_train, maxlen=maxlen)  #Setting the same length of all the comment in train and validation dataset

input_validation = sequence.pad_sequences(X_valid, maxlen=maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
#Get a dict
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embeddings_matrix = np.zeros((nb_words + 1, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)    #the function of embedding_index.get(word) is get the vector of the word.
    if embedding_vector is not None: embeddings_matrix[i] = embedding_vector
#for example, word_index:{'the':1}, embedding_index:{'the': vector}, embedding_matrix:{1: vector}

#line 86-128 one-hot enconding for all aspects in train and validation dataset

Y_train = data.iloc[:, 22:].values
Y_validation = validation.iloc[:, 22:].values

print("model_rcnn")
model = TextClassifier().model(embeddings_matrix, maxlen, word_index, 80)
file_path = model_dir + "model_rcnn_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(file_path, verbose=1, save_weights_only=True)
metrics = Metrics()
callbacks_list = [checkpoint, metrics]
history = model.fit(input_train, Y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(input_validation, Y_validation), callbacks=callbacks_list, verbose=1)
del model1
del history
gc.collect()       #
K.clear_session()

#The basic usage of map see: https://www.runoob.com/python/python-func-map.html -> line 40.
