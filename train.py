import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from keras.layers import Dropout, Dense, LSTM, Embedding
from keras.callbacks import ModelCheckpoint


#load text corpus and name the columns:
data = pd.read_csv('text.txt',sep='|', names=['texts','target'], header = None)
#print(data.shape)
#use subset dataframe for experimenting
#df = data.loc[0:1000]
df = data.sample(1000, axis=0)
#pickle dataset
#df.to_pickle('df.pickle')

#print(df)

#load dataset
#df = pd.read_pickle('df.pickle')

# create the tokenizer: additionally to each words, map [,],.,! and , to numbers
t = Tokenizer(filters='"#$%&()*+-/:;<=>?@\\^_`{|}~\t\n', lower= False) #char_level= True)#maybe try with single characters instead of words

# fit the tokenizer on the(whole) documents
t.fit_on_texts(df.texts + df.target)

# summarize what was learned:
#print(t.word_counts)
#print(t.document_count)
print(t.word_index)
#print(t.word_docs)

#map the words (and characters) to numbers
PREDICTORS = t.texts_to_sequences(df.texts)
TARGET = t.texts_to_sequences(df.target)

#pad all entries of PREDICTORS to the same length:

#find list with maximum length
max_list_predictors = len(max(PREDICTORS,key=len))
#add zeros until each list has same (maximum) length
for i in range(len(PREDICTORS)):
    PREDICTORS[i]= (PREDICTORS[i] + max_list_predictors * [0])[:max_list_predictors]

#todo: target use ofc only one word esel
TARGET = [item[0] for item in TARGET]


#one hot encode the target

TARGET = to_categorical(TARGET)


#split data into training and validation
predictors_train,predictors_test,target_train,target_test = train_test_split(PREDICTORS,TARGET,
train_size=0.9,
test_size=0.1,
random_state=123)

#save them as numpy arrays (?)
predictors_train = np.array(predictors_train)
target_train = np.array(target_train)
predictors_test = np.array(predictors_test)
target_test = np.array(target_test)

print(np.shape(predictors_train),np.shape(target_train))
print(np.shape(predictors_test),np.shape(target_test))


# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, t.word_index.items()))

del data  # free memory

#define the model
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    #model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    # Add Input Embedding Layer
    model.add(Embedding(input_dim = total_words, output_dim=64, input_length=input_len))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    # Add Output Layer
    model.add(Dense(np.shape(target_train)[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


# define the checkpoint
filepath=r"C:\Users\Admin\Desktop\University\Applied Deep Learning\weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#set up model
max_sequence_len=max_list_predictors + 1 #

total_words = len(t.word_index) + 1

model = create_model(max_sequence_len, total_words)
model.summary()


model.fit(predictors_train, target_train, epochs=20,callbacks=callbacks_list, verbose=2)
#
score = model.evaluate(predictors_test, target_test)
print(score)

#

#print(np.shape(predictors_test))
predictions = model.predict(predictors_test)
#print(np.shape(predictions))
#print(np.argmax(predictions))
index = np.argmax(predictions) % np.shape(predictions)[1]

# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, t.word_index.items()))

next_word = reverse_word_map.get(index)
#print(next_word)



def generate_winner(string,model,t,max_length):
    #map string to number sequence
    sequence = t.texts_to_sequences(string)
    #get length of input
    string_length= len(sequence[0])
    # pad the input
    for i in range(len(sequence)):
        sequence[i] = (sequence[i] + max_list_predictors * [0])[:max_list_predictors] #todo: solve use of max_list_predictors better
    #get the right datatype
    sequence = np.array(sequence)
    # make prediction
    predictions = model.predict(sequence)
    # look for the best prediction
    index = np.argmax(predictions) % np.shape(predictions)[1]  # todo: dont use best prediction
    # look up the corresponding word and start building the output
    output_string = reverse_word_map.get(index)
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

#print(generate_winner(['Kümmel, Zucker, Kolbász, Zwiebeln'],model,t,10))


def generate_equal(string,model,t,max_length):
    # map string to number sequence
    sequence = t.texts_to_sequences(string)
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
    # equally chose prediction based on their probability distribution
    index = np.random.choice(range(1,len(predictions)+1),p=predictions) #todo when predictions are higher: p=np.where(predictions<0.1,predictions*0,predictions))
    print(index)
    # look up the corresponding word and start building the output
    output_string = reverse_word_map.get(index)
    for i in range(max_length - 1):
        # append the sequence with the predicted index (word)
        sequence[0, i + string_length] = index
        # make a new prediction
        predictions = model.predict(sequence)
        # get the right data type
        predictions = np.array(predictions[0])
        # equally chose prediction based on their probability distribution
        index = np.random.choice(range(1, len(predictions) + 1),p=predictions)  # todo when predictions are higher: p=np.where(predictions<0.1,predictions*0,predictions))
        # look up the corresponding word
        next_word = reverse_word_map.get(index)
        # append the new word
        output_string = output_string + ' ' + next_word
    return (output_string)


print(generate_equal(['Kümmel, Zucker, Kolbász, Zwiebeln, Ei, Mehl, Salz'],model,t,100))
