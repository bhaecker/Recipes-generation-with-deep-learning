import pickle
import numpy as np
import pandas as pd
import sys

from util import load_data,load_data_ordered,word_mapping,prepare_sets,generate_sets,create_model,train_and_run_experiment

#####
#this script is used for training and experimenting with different slices of the data set

#size of data set
n = 10000

#load data
df = load_data('corpus.txt',n)

#pickle dataset
#df.to_pickle('df_100000.pickle')

#load pickled dataset
#df = pd.read_pickle('df_50080.pickle')

#minimal test sample:
#testcase = {'texts': ['1/2 Eier(n), 100 g Brot, Blaubeeren'],'target': ['Die Eier ( auf das Brot ) legen, dann die Blaubeeren essen.']}
#df = pd.DataFrame(testcase,columns= ['texts', 'target'])
print("done loading data")

#tokenize the input
t,reverse_word_map = word_mapping(df)
t.fit_on_texts(df.texts + df.target)

print('done tokenizing')

#pickle t
pickle.dump(t,open('tokenizer.pickle', 'wb'))

print('done saving tokenizer')

#prepare the data set
PREDICTORS,TARGET = prepare_sets(df,t)

#split the data set
predictors_train,target_train = generate_sets(PREDICTORS,TARGET,1)

#experiment parameters:
max_list_predictors = len(max(PREDICTORS,key=len))
max_sequence_len=max_list_predictors + 1
total_words = len(t.word_index) + 1
output_dim = np.shape(target_train)[1]
neurons = 300
epochs = 1

#create the model
model = create_model(max_sequence_len,total_words,neurons,output_dim)

#train it and run the experiment
model = train_and_run_experiment(model,epochs,predictors_train,target_train)
