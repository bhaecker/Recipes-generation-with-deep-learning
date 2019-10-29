import json
import pandas as pd
from collections import Counter
import re
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.text import text_to_word_sequence
#import textgenrnn

########helper functions:

def unify(x):
  return list(dict.fromkeys(x))

#######

with open('recipes.json') as f: #load data set
  data = json.load(f)
df = pd.DataFrame(data).drop(columns=['Day','Month', 'Name','Url','Weekday','Year'])

#write the data set in one textfile:
concatenated = df['Ingredients'].astype(str) +' '+ df['Instructions']

text_file = open("text.txt", "w", encoding="utf-8")
for line in concatenated:
  line= re.sub(r'\.([a-zA-Z])', r'. \1', line) #put a space after each "."
  line = line.translate({ord("'"): None}) #make a paragraph after each line
  text_file.write(line + "\n")
text_file.close()


#l= df['Ingredients'].tolist()
#Ingredients = []
#for sublist in l:
#    for item in sublist:
#        Ingredients.append(item)

#print(Counter(Ingredients))
#print(len(Ingredients),len(unify(Ingredients)))


#wordlist = df.loc[:, ['Instructions']].stack().tolist()
#words = []
#for sentence in wordlist:
 #   words.append(text_to_word_sequence(sentence,lower=False))

#print(words)

#wordlist = df.loc[:, ['Ingredients']].stack().tolist()
#for ingredientlist in wordlist:
 #   for ingridient in ingredientlist:
  #      words.append(text_to_word_sequence(ingridient,lower=False))

