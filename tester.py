import sys
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from util import generate_testcases,load_data,load_data_ordered,word_mapping,prepare_sets,generate_sets,create_model,train_and_run_experiment,generate_equal,generate_winner,generate_choose_from_n_best,clean_string

####
#this script is used to test the text generation

#number of output words (length of recipe)
words = 50
#for choosing out of n best word specify n
n_bestwords = 100
#number of original (!) ingredient samples for random test generation
m = 50

t = pickle.load(open('tokenizer.pickle', 'rb'))
reverse_word_map = dict(map(reversed, t.word_index.items()))
print(t.word_index)
model = load_model('model.hdf5')
max_list_predictors = model._layers[0].batch_input_shape[1]

string = 'Mehl, Eier, Milch Die Eier mit der Milch verrühren und'
print(clean_string(generate_equal(string,model,t,reverse_word_map,words,max_list_predictors)))

for i in range(2,15):
    string=generate_testcases('recipes.json',m,i)
    print('Test ingredients:',string)
    print('choose equal plus threshold:')
    print(clean_string(generate_equal(string,model,t,reverse_word_map,words,max_list_predictors)))
    print('choose from '+ str(n_bestwords) +' best:')
    print(clean_string(generate_choose_from_n_best(n_bestwords,string,model,t,reverse_word_map,words,max_list_predictors)))
