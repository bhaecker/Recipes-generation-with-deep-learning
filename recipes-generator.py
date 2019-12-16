def start_setup():
    import numpy as np
    from keras.models import load_model
    import pickle
    import sys
    sys.setrecursionlimit(1000000)
    # raw_input returns the empty string for "enter"
    prompt = '> '

    #print("path for tokenizer pls")
    #path_t = input(prompt)
    t = pickle.load(open('tokenizer.pickle', 'rb'))

    #print("path for reverse map")
    #path_reverse = input(prompt)
    reverse_word_map = pickle.load(open('reverse_word_map.pickle', 'rb'))

    #print("done tokenizing")

    #print ("path for model")
    #model_path = input(prompt)
    model = load_model('weights-improvement-10-2.0043.hdf5')

    #print("give max_list_predictors")
    max_list_predictors = 394

    # number of output words (length of recipe)
    print("max. length of recipe:")
    max_length = input(prompt)
    max_length = int(max_length)

    # for choosing out of n best word specify n
    n = 15

    print ("To get started, give an ingridient")
    string = input(prompt)

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

    print("here is your recipie:")
    print(output_string)
start_setup()
