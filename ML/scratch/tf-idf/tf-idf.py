import pandas as pd 
import numpy as np
import re
import math

"""
for td idf we need
 no of words in a single document / row 
 freq of words in every document / row
 need to know whether a word is present in a document / row

"""


df = pd.read_csv("./spam.csv",encoding="latin-1")
row,cols = df.shape

unique={}
len_unique = 0
freq_word_row = {}
word_occ_row = {}
word_row = []
for i in range(row):
    text= df.iloc[i,1]
    text = text.lower()
    text = re.sub(r'[^a-z\s]','',text)
    text = text.split()
    count_words = 0
    for word in text:
        count_words+=1
        if word not in word_occ_row:
            word_occ_row[word]=set()
        word_occ_row[word].add(i)
        if word not in unique:
            unique[word]=len_unique
            len_unique+=1
        if i not in freq_word_row:
            freq_word_row[i]={}
        if word not in freq_word_row[i]:
            freq_word_row[i][word]=0
        freq_word_row[i][word]+=1
    word_row.append(count_words)

def term_freq(i,word,freq):
    total_words = word_row[i]
    return freq/total_words
def inverse_document_freq(word):
    occ_word_rows = len(word_occ_row[word])
   
    return math.log(row/occ_word_rows)
tf_idf =np.zeros((row,len_unique),dtype=float)

for i in range(row):
    text = df.iloc[i,1]
    text = text.lower()
    text = re.sub(r'[^a-z\s]',"",text)
    text = text.split()
    for word in text:
        tf = term_freq(i,word,freq_word_row[i][word])
        idf = inverse_document_freq(word)
        tf_idf[i][unique[word]]=tf*idf
print("finsihed")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = tf_idf
y=df['v1'].map({'spam':1,'ham':0})
x_train , x_test , y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

model = LogisticRegression()

model.fit(x_train,y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test,prediction)

print(accuracy)