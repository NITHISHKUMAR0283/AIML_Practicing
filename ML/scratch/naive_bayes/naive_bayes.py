import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import math
'''
p(spam|free)=p(free|spam)*p(spam)/p(free)

p(free)=count of free / count of all words

p(spam)=count of spam class / count of all class

p(free|spam) = count of free in spam class / count of all  words in spam class


'''
def p_class(label):
     
     Fav_out =class_count[label]
     total_out = 0
     for i in range(len(class_count)):
        total_out+=class_count[i]
     return Fav_out/total_out
def p_word(word):
     w_count = 0
     t_count = 0
     for key,value in class_count.items():
        w_count+=word_count[key][word]
        t_count+=total_word[key]
     return (w_count + 1) / (t_count + len(vocab))
def joint_prob(label,word):
    f_out = word_count[label][word]
    t_out = total_word[label]
    return (f_out + 1) / (t_out + len(vocab))


        
          
def preprocess_text(text):
     text = text.lower()
     text = re.sub(r'[^a-z\s]','',text)
     return text.split()

df = pd.read_csv('./spam.csv',encoding='latin-1')

df=df[['v1','v2']]
df['v1']=df['v1'].map({'ham':0,'spam':1})
df.columns= ['label','text']
train_df, test_df = train_test_split(
    df,
    test_size=0.2
)
rows = len(train_df)


word_count = {0:Counter(),1:Counter()}
class_count = {0:0,1:0}
total_word = {0:0,1:0}
vocab = set()
for i in range(rows):
    text = train_df.iloc[i,1]
    label = train_df.iloc[i,0]
    text = preprocess_text(text)
    for word in text:
        word_count[label][word]+=1
        total_word[label]+=1
        vocab.add(word)
    class_count[label]+=1


def predict(text):
    text= preprocess_text(text)
    scores ={}
    for labels,value in total_word.items():
        p_label = p_class(labels)
        join_pro_word = 0
        for word in text:
            p = joint_prob(labels,word)
            join_pro_word += math.log(p)
        scores[labels] = math.log(p_label) + join_pro_word    
    max_log = max(scores.values())
    ham_exp = math.exp(scores[0] - max_log)
    spam_exp = math.exp(scores[1] - max_log)
    total_exp = ham_exp + spam_exp
    prob_ham = ham_exp / total_exp
    prob_spam = spam_exp / total_exp
    if prob_spam > prob_ham:
        return prob_spam, 1
    return prob_ham, 0
correct = 0
total = len(test_df)
for i in range(total):
    text = test_df.iloc[i,1]
    actual = test_df.iloc[i,0]

    output, predicted = predict(text)

    if predicted == actual:
        correct += 1

accuracy = correct / total
print("Accuracy:", accuracy)