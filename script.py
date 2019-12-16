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

from train_1 import generate_testcases,load_data,load_data_ordered,word_mapping,prepare_sets,generate_sets,create_model,train_and_run_experiment,generate_equal,generate_winner,generate_choose_from_n_best

#todo:attention: generate_testcases has ingridients which are not occurying the dataset, since recipes.json has a lesser structure then the dataset due to the pyriamid scheme
#TODO: maybe preprocess the ingridients and get rid of all ()

######

n = 10000

#load data
#df = load_data_ordered('corpus.txt',n)

#pickle dataset
#df.to_pickle('df_1.pickle')

#load dataset
df = pd.read_pickle('df_2318.pickle')

print(df.texts)

sys.exit()
#minimal test sample
#testcase = {'texts': ['1/2 Eier(n), 100 g Brot, Blaubeeren'],'target': ['Die Eier ( auf das Brot ) legen, dann die Blaubeeren essen.']}
#df = pd.DataFrame(testcase,columns= ['texts', 'target'])

#tokenize the input
#t,reverse_word_map = word_mapping(df)

#pickle t,reverse_word_map
#pickle.dump(t,open('tokenizer.pickle', 'wb'))
#pickle.dump(reverse_word_map,open('reverse_word_map.pickle', 'wb'))


t = pickle.load(open('tokenizer.pickle', 'rb'))
reverse_word_map = pickle.load(open('reverse_word_map.pickle', 'rb'))

print('done tokenizing')

#prepare the data set
#PREDICTORS,TARGET = prepare_sets(df)

#split the data set
#predictors_train,target_train,predictors_test,target_test = generate_sets(PREDICTORS,TARGET,1)

#experiment parameters:  (solve better)
max_list_predictors =394# len(max(PREDICTORS,key=len))
#print('remember this number and hardcode it:',max_list_predictors) #394
#max_sequence_len=max_list_predictors + 1
#total_words = len(t.word_index) + 1
#output_dim = np.shape(target_train)[1]
neurons = 200
epochs = 20

#create the model
#model = create_model(max_sequence_len,total_words,neurons,output_dim)

#train it and run the experiment
#model = train_and_run_experiment(model,epochs,predictors_train,target_train,predictors_test,target_test,t)

#load model
model = load_model('weights-improvement-09-2.8373.hdf5')


#number of output words (length of recipe)
words = 20
#for choosing out of n best word specify n
n_bestwords = 15
#number of original (!) ingridient samples for random test generation
m = 5


print('choose from n best:')
print(generate_choose_from_n_best(n_bestwords,generate_testcases('recipes.json',m,5),model,t,reverse_word_map,words,max_list_predictors))
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
