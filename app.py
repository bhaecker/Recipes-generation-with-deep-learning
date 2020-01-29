from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
import os
    # disable debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import load_model
from keras import backend as K
import pickle
import math
import re
global graph
import tensorflow as tf
import sys
graph = tf.get_default_graph()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    t = pickle.load(open('tokenizer.pickle', 'rb'))
   
    reverse_word_map = dict(map(reversed, t.word_index.items()))
    # reverse_word_map = pickle.load(open('reverse_word_map.pickle', 'rb'))

    # load the model and get the input dimensions
    model = load_model('model.hdf5', compile=False)
    #model._make_predict_function()

    max_list_predictors = model._layers[0].batch_input_shape[1]

    if request.method == 'POST':
        message = request.form['message']
        data = [message]

        max_length = request.form['length']
        max_length = int(float(max_length))


        # number of output words (length of recipe)
        #max_length = 60

        # map string to number sequence
        sequence = t.texts_to_sequences(data)  # !!!!
        # get length of input
        string_length = len(sequence[0])
        # pad the input
        for i in range(len(sequence)):
            sequence[i] = (sequence[i] + max_list_predictors * [0])[
                          :max_list_predictors]
        # get the right datatype

        sequence = np.array(sequence)
        # make prediction
        #with graph.as_default():
        predictions = model.predict(sequence)
        K.clear_session()
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
        # look up the corresponding word and start building the output
        output_string = reverse_word_map.get(index)
        # predict following words on input sequence plus the already predicted words:
        for i in range(max_length - 1):
            # append the sequence with the predicted index (word)
            sequence[0, i + string_length] = index
            # make a new prediction
            #with graph.as_default():
            model = load_model('model.hdf5', compile=False)
            predictions = model.predict(sequence)
            K.clear_session()
            # equally choose prediction based on their probability distribution:
            # define threshold to omit some predictions
            threshold = np.average(predictions)
            # set all predictions under the threshold to zero
            p = np.where(predictions < threshold, predictions * 0, predictions)
            # make p a proper probability distribution again
            p /= np.sum(p)
            p = p[0]
            # chose the index randomly with respect to the likelyhood of getting chosen i.e. the probability distribution
            index = np.random.choice(np.arange(len(predictions[0])), p=p)
            # look up the corresponding word
            next_word = reverse_word_map.get(index)
            # append the new word
            output_string = output_string + ' ' + next_word

        # clean the string by making it german sentences:
        # regex for finding . , ! and ?
        regex = r"\s*([.,!?])\s*"
        # replacing regex with nothing
        output_string = re.sub(regex, "\\1 ", output_string)
        # "[regex] matches the start of the string ^ or .?! followed by optional spaces" (Psidom stackoverfow)
        regex = "(^|[.?!])\s*([a-zA-Z])"
        # "use lambda function to convert the captured group to upper case" (Psidom stackoverfow)
        output_string = re.sub(regex, lambda p: p.group(0).upper(), output_string)
        # find last occurence of . ! or ?
        k = max(output_string.rfind("."), output_string.rfind("!"), output_string.rfind("?"))
        # cut the string afterwards
        if k != -1:
            output_string = output_string[:k + 1]
        # output the string
    return render_template('result.html', prediction=output_string)

if __name__ == '__main__':
    app.run(debug=True)