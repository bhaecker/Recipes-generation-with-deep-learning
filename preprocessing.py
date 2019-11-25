import json
import numpy as np
import pandas as pd
from collections import Counter
import re
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.text import text_to_word_sequence
#import textgenrnn

with open('recipes.json') as f: #load data set
  data = json.load(f)
df = pd.DataFrame(data).drop(columns=['Day','Month', 'Name','Url','Weekday','Year'])

df['Ingredients'] = df['Ingredients'].astype(str) #convert Ingredients list to a single string
#print(type(df['Ingredients'][0])) #check if worked

#!!todo: put spaces before ] and after [

for i in range(0,len(df['Ingredients'])):
  df.at[i,'Ingredients'] = re.sub(r'\.([a-zA-Z])', r'. \1', df['Ingredients'][i]) #put a space after each "."

for i in range(0,len(df['Instructions'])):
  df.at[i,'Instructions'] = re.sub(r'\.([a-zA-Z])', r'. \1', df['Instructions'][i]) #put a space after each "."

#at the end pls
#df.Ingredients = df.Ingredients.str.pad(width=df.Ingredients.map(len).max(), side='right') #pad all entries to same length
#df.Instructions = df.Instructions.str.pad(width=df.Instructions.map(len).max(), side='right')

#size = 0
#for index, row in df.iterrows():
 # size = size + len(row[1].split())

#print(size)


#text_file = open("text.txt", "w", encoding="utf-8")

for index,row in df.iterrows():
  split_list = row[1].split()
  for place in range(0,len(split_list)):
    line = row[0] + ' '
    for i in range(0,place):
      line = line + split_list[i] + ' '
    line = line + ' | ' #seperates training and target strings
    for j in range(place,len(split_list)):
      line = line + split_list[j] + ' '
    line = line.translate({ord("'"): None})  # make a paragraph after each line
    text_file.write(line + "\n")

print('done')









#wrong dataframe creation

#for index,row in df.iterrows():
 # split_list = row[1].split()
  #line = row[0] +'|'+split_list[0]

  #line = line.translate({ord("'"): None})  # make a paragraph after each line
  #text_file.write(line + "\n")

  #for word in split_list[1:-1]:
   # line = line+' '+word
    #line = line.translate({ord("'"): None})  # make a paragraph after each line
    #text_file.write(line + "\n")

#print('done')


#slow dataframe creation
#determien size of new dataframe
#size = 0
#for index, row in df.iterrows():
 # size = size + len(row[1].split())

#print(size)

#ds = pd.DataFrame(index=np.arange(size),columns=['Ingredients', 'Instructions'])



#i=0
#for index, row in df.iterrows():
 # split_list = row[1].split()

  #ds.at[i, 'Instructions'] = split_list[0]

  #for word in split_list[1:-1]:
   # ds.at[i+1,'Instructions'] = ds.Instructions[i]+' '+word
    #i = i+1

  #print(i/len(ds['Instructions']),'%')

#print(ds)
#print(ds["Instructions"].iloc[-1], test_list)

#write the data set in one textfile:
#concatenated = df['Ingredients'].astype(str) +' '+ df['Instructions']

#text_file = open("text.txt", "w", encoding="utf-8")
#for line in concatenated:
 # line= re.sub(r'\.([a-zA-Z])', r'. \1', line) #put a space after each "."
  #line = line.translate({ord("'"): None}) #make a paragraph after each line
  #text_file.write(line + "\n")
#text_file.close()


##########helper functions:

#def unify(x):
 # return list(dict.fromkeys(x))

#########



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

