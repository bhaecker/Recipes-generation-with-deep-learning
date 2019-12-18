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

from train_1 import generate_testcases,load_data,load_data_ordered,word_mapping,prepare_sets,generate_sets,create_model,train_and_run_experiment,generate_equal,generate_winner,generate_choose_from_n_best,clean_string

######
#this script is used for training and experimenting with different slices of the data set

#size of data set
n = 50080

#load data
#df = load_data('corpus.txt',n)

#pickle dataset
#df.to_pickle('df_50000_uno.pickle')

#load pickled dataset
df = pd.read_pickle('df_50080.pickle')

#minimal test sample:
#testcase = {'texts': ['1/2 Eier(n), 100 g Brot, Blaubeeren'],'target': ['Die Eier ( auf das Brot ) legen, dann die Blaubeeren essen.']}
#df = pd.DataFrame(testcase,columns= ['texts', 'target'])
print("done loading data")

#tokenize the input
t,reverse_word_map = word_mapping(df)
print('done tokenizing')

#pickle t,reverse_word_map
pickle.dump(t,open('tokenizer_50080.pickle', 'wb'))
#pickle.dump(reverse_word_map,open('reverse_word_map.pickle', 'wb'))
print('done saving tokenizer')

#prepare the data set
PREDICTORS,TARGET = prepare_sets(df)

#split the data set
predictors_train,target_train,predictors_test,target_test = generate_sets(PREDICTORS,TARGET,0.99)

#experiment parameters:
max_list_predictors = len(max(PREDICTORS,key=len))
max_sequence_len=max_list_predictors + 1
total_words = len(t.word_index) + 1
output_dim = np.shape(target_train)[1]
neurons = 300
epochs = 20

#create the model
model = create_model(max_sequence_len,total_words,neurons,output_dim)

#train it and run the experiment
model = train_and_run_experiment(model,epochs,predictors_train,target_train,predictors_test,target_test,t)

#number of output words (length of recipe)
words = 3
#for choosing out of n best word specify n
n_bestwords = 33
#number of original (!) ingridient samples for random test generation
m = 20

#t = pickle.load(open('tokenizer_200000.pickle', 'rb'))
#reverse_word_map = dict(map(reversed, t.word_index.items()))

#model = load_model('weights-improvement-02-9.1294.hdf5',compile=False)
#max_list_predictors = model._layers[0].batch_input_shape[1]


print('choose from n best:')
print(clean_string(generate_choose_from_n_best(n_bestwords,generate_testcases('recipes.json',m,5),model,t,reverse_word_map,words,max_list_predictors)))
print(generate_choose_from_n_best(n_bestwords,generate_testcases('recipes.json',m,10),model,t,reverse_word_map,words,max_list_predictors))
print(generate_choose_from_n_best(n_bestwords,generate_testcases('recipes.json',m,20),model,t,reverse_word_map,words,max_list_predictors))

print('choose equal plus threshold:')
print(generate_equal(generate_testcases('recipes.json',m,5),model,t,reverse_word_map,words,max_list_predictors))
print(generate_equal(generate_testcases('recipes.json',m,10),model,t,reverse_word_map,words,max_list_predictors))
print(generate_equal(generate_testcases('recipes.json',m,20),model,t,reverse_word_map,words,max_list_predictors))

print('choose winner:')
print(generate_winner(generate_testcases('recipes.json',m,5),model,t,reverse_word_map,words,max_list_predictors))
print(generate_winner(generate_testcases('recipes.json',m,10),model,t,reverse_word_map,words,max_list_predictors))
print(generate_winner(generate_testcases('recipes.json',m,20),model,t,reverse_word_map,words,max_list_predictors))
