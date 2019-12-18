def start_setup():
    'is used to create the .exe file, which relies on a trained model, the corresponding tokenizer and the users input'
    import numpy as np
    from keras.models import load_model
    import pickle
    import math
    import re
    #only needed in the spec file while creating the exe file via pyinstaller --onefile recipes-generator.py. The predictions are made from the 33 most likely predicted words
    #import sys
    #sys.setrecursionlimit(1000000)
    prompt = '> '
    print("Welcome to the recipe generator!")
    print("This generator uses a RNN Model trained on recipes of chefkoch.de, to generate recipes in german.")
    print("Please make sure the tokenizer.pickle and model.hdf5 files are in the same filepath as the .exe file you just called.")
    #load the tokenizer and build the reverse mapping
    t = pickle.load(open('tokenizer.pickle', 'rb'))
    reverse_word_map = dict(map(reversed, t.word_index.items()))
    #reverse_word_map = pickle.load(open('reverse_word_map.pickle', 'rb'))

    #load the model and get the input dimensions
    model = load_model('model.hdf5',compile=False)
    max_list_predictors = model._layers[0].batch_input_shape[1]

    print ("To get started, give the ingredients you want to use, separated by a comma:")
    string = input(prompt)

    # number of output words (length of recipe)
    print("Please specify the maximal length you want your recipe to have:")
    print("(Note, that the time to generate the recipe depends on the length of it.)")
    max_length = int(input(prompt))

    # for choosing out of n best word specify n
    n = 33
    print("While the recipe is generated, you can start collecting all the ingredients :)")
    # map string to number sequence
    sequence = t.texts_to_sequences(string)
    # get length of input
    string_length = len(sequence[0])
    # pad the input
    for i in range(len(sequence)):
        sequence[i] = (sequence[i] + max_list_predictors * [0])[
                      :max_list_predictors]
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
        if i == math.ceil(max_length/2):
            print("We are halfway there!")
            
    #clean the string by making it german sentences:
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
    print("Here is your recipe:")
    print(output_string)

start_setup()
