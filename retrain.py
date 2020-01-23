import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from keras.layers import Dropout, Dense, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from util import prepare_sets_retraining,generate_testcases,load_data,load_data_ordered,word_mapping,prepare_sets,generate_sets,create_model,train_and_run_experiment

######
#this script is used for retraining an existing model

#load model to retrain
model = load_model('model.hdf5')
model.summary()

#size of data set for retraining
n = 1000

#load data for retraining
#df = load_data('corpus.txt',n)

#or use existing data set
df = pd.read_pickle('df_10k.pickle')

#pickle dataset
#df.to_pickle('df_100000.pickle')

print("done loading data")

#load original tokenizer
t = pickle.load(open('tokenizer.pickle', 'rb'))
reverse_word_map = dict(map(reversed, t.word_index.items()))

print('done loading tokenizer')

#prepare the data set
PREDICTORS,TARGET = prepare_sets_retraining(df,t,model)

#split the data set
predictors_train,target_train = generate_sets(PREDICTORS,TARGET,1)

#experiment parameters:
max_list_predictors = model._layers[0].batch_input_shape[1]
max_sequence_len=max_list_predictors + 1
total_words = len(t.word_index) + 1
output_dim = np.shape(target_train)[1]
epochs = 2

#train it and run the experiment
model = train_and_run_experiment(model,epochs,predictors_train,target_train)

