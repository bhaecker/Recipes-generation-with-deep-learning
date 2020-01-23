import sys
import json
import random as rd
from random import randint
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dropout, Dense, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

####
#functions for preparing the data sets, test instances and predictions:

def generate_testcases(data_path,size,length):
    'generates test strings (list of ingredients) from original jason file'
    # load data set
    with open(data_path) as f:
        data = json.load(f)
    # safe as dataframe and drop columns which are not needed
    df = pd.DataFrame(data).drop(columns=['Instructions','Day', 'Month', 'Name', 'Url', 'Weekday', 'Year'])
    # slize it to the right size - no random selection
    df = df.loc[0:size]
    #get the row size
    row_size = len(df.index)
    #empty list for output
    list= []
    #put 'length' items in list
    for i in range(length):
        #choose random cell
        cell = df[df.columns[0]][randint(0, row_size - 1)]
        #get random ingredient
        next_item = rd.sample(cell, k=1)[0]
        #check if ingredient is already in the list
        if next_item in list:
            continue
        #put ingredient in the list if not
        list = list + [next_item + ',']
    list = " ".join(list)
    return(list)


def load_data(data_path,size):
    'creates a data set by randomly choosing samples'
    data = pd.read_csv(data_path, sep='|', names=['texts', 'target'], header=None)
    #random picking size-many samples
    df = data.sample(size, axis=0)
    # get rid of [ and ]
    df['texts'] = df['texts'].replace({'\[':''}, regex=True)
    df['texts'] = df['texts'].replace({'\]': ''}, regex=True)
    del data # free memory
    return(df)

def load_data_ordered(data_path,size):
    'creates a data set by choosing samples in order from the original file'
    data = pd.read_csv(data_path, sep='|', names=['texts', 'target'], header=None)
    # use subset dataframe for experimenting
    df = data.loc[0:size] #no random selection
    #get rid of [ and ]
    df['texts'] = df['texts'].replace({'\[': ''}, regex=True)
    df['texts'] = df['texts'].replace({'\]': ''}, regex=True)
    del data # free memory
    return(df)


def word_mapping(df):
    'determine a mapping from words to numbers and the inverse operation'
    df['texts'] = df['texts'].str.replace(',', '')
    t = Tokenizer(num_words= 1000,filters='"#$%&*+-:;<=>?@\\^_`{|}~\t\n',
                  lower=False)
    t.fit_on_texts(df.texts +' '+df.target)
    #summarize what was learned:
    print(t.word_counts)
    print(t.document_count)
    print(t.word_index)
    print(t.word_docs)
    #create a reverse word map
    reverse_word_map = dict(map(reversed, t.word_index.items()))
    return(t,reverse_word_map)

def prepare_sets(df,t):
    'prepare the data set for the machine learning task by mapping all words to numbers and one-hot-encoding the target'

    # map the words (and characters) to numbers
    PREDICTORS = t.texts_to_sequences(df.texts)
    TARGET = t.texts_to_sequences(df.target)
    # pad all entries of PREDICTORS to the same length:
    # find list with maximum length
    max_list_predictors = len(max(PREDICTORS, key=len))
    # add zeros until each list has same (maximum) length
    for i in range(len(PREDICTORS)):
        PREDICTORS[i] = (PREDICTORS[i] + max_list_predictors * [0])[:max_list_predictors]
    #target is only one word i.e. we predict only the next word and not all the next words
    for i in range(0,len(TARGET)):
        if len(TARGET[i]) == 0:
            TARGET[i] = 0
        else:
            TARGET[i] = TARGET[i][0]
    # one hot encode the target
    TARGET = to_categorical(TARGET)
    return(PREDICTORS,TARGET)

def prepare_sets_retraining(df,t,model):
    'prepare the data set for retraining an existing model'
    # map the words (and characters) to numbers
    PREDICTORS = t.texts_to_sequences(df.texts)
    TARGET = t.texts_to_sequences(df.target)
    # pad all entries of PREDICTORS to the same length:
    max_list_predictors = model._layers[0].batch_input_shape[1]
    #if max_list_predictors > len(PREDICTORS):

    # add zeros until each list has same (maximum) length
    for i in range(len(PREDICTORS)):
        PREDICTORS[i] = (PREDICTORS[i] + max_list_predictors * [0])[:max_list_predictors]
    #target is only one word i.e. we predict only the next word and not all the next words
    for i in range(0,len(TARGET)):
        if len(TARGET[i]) == 0:
            TARGET[i] = 0
        else:
            TARGET[i] = TARGET[i][0]
    # one hot encode the target
    output_dimension = model.get_layer('dense_3').output.shape[1]
    print(TARGET)
    TARGET = to_categorical(TARGET, num_classes=output_dimension)
    return(PREDICTORS,TARGET)


def generate_sets(PREDICTORS,TARGET,train_size):
    'split the sets for training'
    if train_size == 1:
        predictors_train = np.array(PREDICTORS)
        target_train = np.array(TARGET)
        return(predictors_train,target_train)
    # split data into training and validation
    predictors_train, predictors_test, target_train, target_test = train_test_split(PREDICTORS, TARGET,
                                                                                    train_size=train_size,random_state=123)

    # save them as numpy arrays
    predictors_train = np.array(predictors_train)
    target_train = np.array(target_train)
    predictors_test = np.array(predictors_test)
    target_test = np.array(target_test)
    return(predictors_train,target_train,predictors_test,target_test)


##functions for creating and training the DNN model:

def create_model(max_sequence_len,total_words,neurons,output_dim):
    'sequential model with LSTM and dropout layers'
    input_len = max_sequence_len - 1
    model = Sequential()
    # Add Input Embedding Layer
    model.add(Embedding(input_dim=total_words, output_dim=128, input_length=input_len))
    model.add(LSTM(neurons, return_sequences=True))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(0.3))
    model.add(LSTM(neurons))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(0.2))
    # Add Output Layer
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return(model)

def train_and_run_experiment(model,epochs,predictors_train,target_train):
    'train the model and save it for each epoch'
    # define the checkpoint
    filepath=r"C:\Users\Admin\Desktop\University\Applied Deep Learning\weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    #train model on data
    model.fit(predictors_train, target_train, epochs=epochs, batch_size=20, verbose=1,callbacks=callbacks_list)
    return(model)


##functions for generating the predictions and cleaning the output string:

def generate_winner(string,model,t,reverse_word_map,max_length,max_list_predictors):
    'generate text while chosing always the most likely next word'
    string = string.replace(',', '')
    #map string to number sequence
    sequence = t.texts_to_sequences([string])
    #get length of input
    string_length= len(sequence[0])
    # pad the input
    for i in range(len(sequence)):
        sequence[i] = (sequence[i] + max_list_predictors * [0])[:max_list_predictors]
    #get the right datatype
    sequence = np.array(sequence)
    # make prediction
    predictions = model.predict(sequence)
    # look for the best prediction
    index = np.argmax(predictions)
    # look up the corresponding word and start building the output
    output_string = reverse_word_map.get(index)
    # predict following words on input sequence plus the already predicted words:
    for i in range(max_length-1):
        #append the sequence with the predicted index (word)
        sequence[0,i+string_length] = index
        #make a new prediction
        predictions = model.predict(sequence)
        #look for the best prediction
        index = np.argmax(predictions) #% np.shape(predictions)[1]
        #look up the corresponding word
        next_word = reverse_word_map.get(index)
        #append the new word
        output_string = output_string +' '+ next_word
    return(output_string)


def generate_equal(string,model,t,reverse_word_map,max_length,max_list_predictors):
    'generate string, where each word is chosen according to the probability of its prediction, if predcition is above mean prediction'
    # map string to number sequence
    sequence = t.texts_to_sequences([string])
    # get length of input
    string_length = len(sequence[0])
    # pad the input
    for i in range(len(sequence)):
        sequence[i] = (sequence[i] + max_list_predictors * [0])[
                      :max_list_predictors]  # todo: solve use of max_list_predictors better
    # get the right datatype
    sequence = np.array(sequence)
    # make prediction
    predictions = model.predict(sequence)
    # get the right data type
    predictions = np.array(predictions[0])
    #equally choose prediction based on their probability distribution:
    #define threshold to omit some predictions
    threshold = np.average(predictions)
    #set all predictions under the threshold to zero
    p = np.where(predictions < threshold, predictions * 0, predictions)
    #make p a proper probability distribution again
    p /= np.sum(p)
    #chose the index randomly with respect to the likelyhood of getting chosen i.e. the probability distribution
    index = np.random.choice(np.arange(len(predictions)),p = p)
    # look up the corresponding word and start building the output
    output_string = reverse_word_map.get(index)
    #predict following words on input sequence plus the already predicted words:
    for i in range(max_length - 1):
        # append the sequence with the predicted index (word)
        sequence[0, i + string_length] = index
        # make a new prediction
        predictions = model.predict(sequence)
        # get the right data type
        predictions = np.array(predictions[0])
        # equally choose prediction based on their probability distribution:
        # define threshold to omit some predictions
        threshold = np.average(predictions)
        # set all predictions under the threshold to zero
        p = np.where(predictions < threshold, predictions * 0, predictions)
        # make p a proper probability distribution again
        p /= np.sum(p)
        # chose the index randomly with respect to the likelyhood of getting chosen i.e. the probability distribution
        index = np.random.choice(np.arange(len(predictions)), p=p)
        # look up the corresponding word
        next_word = reverse_word_map.get(index)
        # append the new word
        output_string = output_string + ' ' + next_word
    return (output_string)

def generate_choose_from_n_best(n,string,model,t,reverse_word_map,max_length,max_list_predictors):
    'generate string, where the next word is chosen only from the n most likely words'
    # map string to number sequence
    sequence = t.texts_to_sequences([string])
    # get length of input
    string_length = len(sequence[0])
    # pad the input
    for i in range(len(sequence)):
        sequence[i] = (sequence[i] + max_list_predictors * [0])[
                      :max_list_predictors]  # todo: solve use of max_list_predictors better
    # get the right datatype
    sequence = np.array(sequence)
    # make prediction
    predictions = model.predict(sequence)
    # get the right data type
    predictions = np.array(predictions[0])
    # equally chose n best prediction based on their probability distribution
    #get indicies of n largest values
    n_indicies = predictions.argsort()[-n:][::-1]
    #get the indices of the not n largest values
    not_n_indices = np.setxor1d(np.indices(predictions.shape), n_indicies)
    #set values of not n largest values to zero
    predictions[not_n_indices] = 0
    # make predictions a proper probability distribution again
    predictions /= np.sum(predictions)
    # chose the index randomly with respect to the likelyhood of getting chosen i.e. the probability distribution
    index = np.random.choice(np.arange(len(predictions)), p=predictions)
    # look up the corresponding word and start building the output
    output_string = reverse_word_map.get(index)
    # predict following words on input sequence plus the already predicted words:
    for i in range(max_length - 1):
        # append the sequence with the predicted index (word)
        sequence[0, i + string_length] = index
        # make a new prediction
        predictions = model.predict(sequence)
        # get the right data type
        predictions = np.array(predictions[0])
        # equally chose n best prediction based on their probability distribution
        # get indicies of n largest values
        n_indicies = predictions.argsort()[-n:][::-1]
        # get the indices of the not n largest values
        not_n_indices = np.setxor1d(np.indices(predictions.shape), n_indicies)
        # set values of not n largest values to zero
        predictions[not_n_indices] = 0
        # make predictions a proper probability distribution again
        predictions /= np.sum(predictions)
        # chose the index randomly with respect to the likelyhood of getting chosen i.e. the probability distribution
        index = np.random.choice(np.arange(len(predictions)), p=predictions)
        # look up the corresponding word
        next_word = reverse_word_map.get(index)
        # append the new word
        output_string = output_string + ' ' + next_word
    return(output_string)


def clean_string(string):
    'changes a string into german sentences'
    #regex for finding . , ! and ?
    regex = r"\s*([.,!?])\s*"
    #replacing regex with nothing
    string = re.sub(regex, "\\1 ", string)
    #"[regex] matches the start of the string ^ or .?! followed by optional spaces" (Psidom stackoverfow)
    regex = "(^|[.?!])\s*([a-zA-Z])"
    #"use lambda function to convert the captured group to upper case" (Psidom stackoverfow)
    string = re.sub(regex, lambda p: p.group(0).upper(), string)
    #find last occurence of . ! or ?
    k = max(string.rfind("."), string.rfind("!"), string.rfind("?"))
    # cut the string afterwards
    if k != -1:
        string = string[:k + 1]
    #output the string
    return(string)
