import json
import numpy as np
import pandas as pd
from collections import Counter
import re

#preprocess the raw json file

#load data set
with open('recipes.json') as f:
  data = json.load(f)
#safe as dataframe and drop columns which are not needed
df = pd.DataFrame(data).drop(columns=['Day','Month', 'Name','Url','Weekday','Year'])

#convert Ingredients list to a single string
df['Ingredients'] = df['Ingredients'].astype(str)

for i in range(0,len(df['Ingredients'])):

  df.at[i,'Ingredients'] = re.sub(r'\.([a-zA-Z])', r'. \1', df['Ingredients'][i])  #put a space after each "."
  df.at[i,'Ingredients'] = re.sub(r'\!([a-zA-Z])', r'! \1', df['Ingredients'][i])  # put a space after each "!"
  df.at[i,'Ingredients'] = re.sub(r'\,([a-zA-Z])', r', \1', df['Ingredients'][i])  # put a space after each ","
  df.at[i,'Ingredients'] = re.sub(r'\:([a-zA-Z])', r': \1', df['Ingredients'][i])  # put a space after each ":"
  df.at[i,'Ingredients'] = re.sub(r'\)([a-zA-Z])', r') \1', df['Ingredients'][i])  # put a space after each ")"
  df.at[i,'Ingredients'] = re.sub(r'\(([a-zA-Z])', r'( \1', df['Ingredients'][i])  # put a space after each "("
  df.at[i,'Ingredients'] = re.sub(r'\[([a-zA-Z])', r'[ \1', df['Ingredients'][i])  # put a space after each "["
  df.at[i,'Ingredients'] = re.sub(r'\]([a-zA-Z])', r'] \1', df['Ingredients'][i])  # put a space after each "]"

  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\.', r'\1 .', df['Ingredients'][i])  # put a space before each "."
  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\,', r'\1 ,', df['Ingredients'][i])  # put a space before each ","
  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\!', r'\1 !', df['Ingredients'][i])  # put a space before each "!"
  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\:', r'\1 :', df['Ingredients'][i])  # put a space before each ":"
  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\)', r'\1 )', df['Ingredients'][i])  # put a space before each ")"
  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\(', r'\1 (', df['Ingredients'][i])  # put a space before each "("
  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\[', r'\1 [', df['Ingredients'][i])  # put a space before each "["
  df.at[i, 'Ingredients'] = re.sub(r'([a-zA-Z])\]', r'\1 ]', df['Ingredients'][i])  # put a space before each "]"

for i in range(0,len(df['Instructions'])):

  df.at[i,'Instructions'] = re.sub(r'\.([a-zA-Z])', r'. \1', df['Instructions'][i]) #put a space after each "."
  df.at[i,'Instructions'] = re.sub(r'\!([a-zA-Z])', r'! \1', df['Instructions'][i])  # put a space after each "!"
  df.at[i,'Instructions'] = re.sub(r'\,([a-zA-Z])', r', \1', df['Instructions'][i])  # put a space after each ","
  df.at[i,'Instructions'] = re.sub(r'\:([a-zA-Z])', r': \1', df['Instructions'][i])  # put a space after each ":"
  df.at[i,'Instructions'] = re.sub(r'\)([a-zA-Z])', r') \1', df['Instructions'][i])  # put a space after each ")"
  df.at[i,'Instructions'] = re.sub(r'\(([a-zA-Z])', r'( \1', df['Instructions'][i])  # put a space after each "("
  df.at[i,'Instructions'] = re.sub(r'\[([a-zA-Z])', r'[ \1', df['Instructions'][i])  # put a space after each "["
  df.at[i,'Instructions'] = re.sub(r'\]([a-zA-Z])', r'] \1', df['Instructions'][i])  # put a space after each "]"

  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\.', r'\1 .', df['Instructions'][i])  #put a space before each "."
  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\,', r'\1 ,', df['Instructions'][i])  # put a space before each ","
  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\!', r'\1 !', df['Instructions'][i])  # put a space before each "!"
  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\:', r'\1 :', df['Instructions'][i])  # put a space before each ":"
  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\)', r'\1 )', df['Instructions'][i])  # put a space before each ")"
  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\(', r'\1 (', df['Instructions'][i])  # put a space before each "("
  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\[', r'\1 [', df['Instructions'][i])  # put a space before each "["
  df.at[i,'Instructions'] = re.sub(r'([a-zA-Z])\]', r'\1 ]', df['Instructions'][i])  # put a space before each "]"

#open text file to save the result
#text_file = open("corpus.txt", "w", encoding="utf-8")

#slice each sample into many
for index,row in df.iterrows():
  #make a list of words out of Instructions
  split_list = row[1].split()
  #make number-of-words many smaples out of one sample
  for place in range(0,len(split_list)):
    #start with Ingredients as predictors
    line = row[0] + ' '
    #append the line with predictors/trainig strings
    for i in range(0,place):
      line = line + split_list[i] + ' '
    #seperate training and target strings
    line = line + ' | '
    #append the line with target
    for j in range(place,len(split_list)):
      line = line + split_list[j] + ' '
    # make a paragraph after each line
    line = line.translate({ord("'"): None})
    text_file.write(line + "\n")

print('done')
