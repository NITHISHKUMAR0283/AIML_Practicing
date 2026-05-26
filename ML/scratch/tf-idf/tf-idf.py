import pandas as pd
import numpy as np
import re
import math

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("./spam.csv", encoding="latin-1")

row, cols = df.shape

# =========================
# TF-IDF STORAGE
# =========================

unique = {}
len_unique = 0

freq_word_row = {}
word_occ_row = {}

word_row = []

# =========================
# BUILD VOCABULARY
# =========================

for i in range(row):

    text = str(df['TEXT'][i])

    text = text.lower()

    # preserve important phishing patterns
    text = re.sub(r'http\S+', ' URLTOKEN ', text)
    text = re.sub(r'\S+@\S+', ' EMAILTOKEN ', text)
    text = re.sub(r'\d+', ' PHONETOKEN ', text)

    # cleanup
    text = re.sub(r'[^a-z\s]', '', text)

    text = text.split()

    count_words = 0

    for word in text:

        count_words += 1

        # document frequency
        if word not in word_occ_row:
            word_occ_row[word] = set()

        word_occ_row[word].add(i)

        # vocabulary indexing
        if word not in unique:
            unique[word] = len_unique
            len_unique += 1

        # frequency inside current document
        if i not in freq_word_row:
            freq_word_row[i] = {}

        if word not in freq_word_row[i]:
            freq_word_row[i][word] = 0

        freq_word_row[i][word] += 1

    word_row.append(count_words)

# =========================
# TF FUNCTION
# =========================

def term_freq(i, word, freq):

    total_words = word_row[i]

    return freq / total_words

# =========================
# IDF FUNCTION
# =========================

def inverse_document_freq(word):

    occ_word_rows = len(word_occ_row[word])

    return math.log(row / occ_word_rows)

# =========================
# TF-IDF MATRIX
# =========================

tf_idf = np.zeros((row, len_unique), dtype=float)

for i in range(row):

    text = str(df['TEXT'][i])

    text = text.lower()

    # preserve phishing patterns
    text = re.sub(r'http\S+', ' URLTOKEN ', text)
    text = re.sub(r'\S+@\S+', ' EMAILTOKEN ', text)
    text = re.sub(r'\d+', ' PHONETOKEN ', text)

    text = re.sub(r'[^a-z\s]', '', text)

    text = text.split()

    for word in text:

        tf = term_freq(i, word, freq_word_row[i][word])

        idf = inverse_document_freq(word)

        tf_idf[i][unique[word]] = tf * idf

print("TF-IDF Finished")

# =========================
# EXTRA METADATA FEATURES
# =========================

extra_features = np.zeros((row, 3), dtype=float)

for i in range(row):

    # URL
    if str(df['URL'][i]).lower() == "yes":
        extra_features[i][0] = 1

    # EMAIL
    if str(df['EMAIL'][i]).lower() == "yes":
        extra_features[i][1] = 1

    # PHONE
    if str(df['PHONE'][i]).lower() == "yes":
        extra_features[i][2] = 1

# =========================
# FINAL FEATURE MATRIX
# =========================

X = np.hstack((tf_idf, extra_features))

# =========================
# LABELS
# =========================

y = df['LABEL'].map({
    'ham': 0,
    'spam': 1,
    'smishing': 2
})

# =========================
# TRAIN TEST SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

# =========================
# TEST ACCURACY
# =========================

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)


test_text = "hi can you forward me the pin number which is present before the atm card"

original_test_text = test_text


test_text = test_text.lower()

test_text = re.sub(r'http\S+', ' URLTOKEN ', test_text)
test_text = re.sub(r'\S+@\S+', ' EMAILTOKEN ', test_text)
test_text = re.sub(r'\d+', ' PHONETOKEN ', test_text)

test_text = re.sub(r'[^a-z\s]', '', test_text)

test_text = test_text.split()



test_vector = np.zeros((1, len_unique + 3), dtype=float)



test_freq = {}

total_words = 0

for word in test_text:

    total_words += 1

    if word not in test_freq:
        test_freq[word] = 0

    test_freq[word] += 1


for word in test_freq:


    if word not in unique:
        continue

    tf = test_freq[word] / total_words

    if word in word_occ_row:
        idf = math.log(row / len(word_occ_row[word]))
    else:
        idf = 0

    test_vector[0][unique[word]] = tf * idf


if "http" in original_test_text.lower():
    test_vector[0][len_unique] = 1


if "@" in original_test_text:
    test_vector[0][len_unique + 1] = 1


if any(char.isdigit() for char in original_test_text):
    test_vector[0][len_unique + 2] = 1


prediction = model.predict(test_vector)



if prediction[0] == 0:
    print("HAM")

elif prediction[0] == 1:
    print("SPAM")

else:
    print("SMISHING")




from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =========================
# REDUCE DIMENSIONS
# =========================

pca = PCA(n_components=2)

X_2d = pca.fit_transform(X)

# =========================
# TRAIN NEW MODEL FOR VISUALIZATION
# =========================

visual_model = LogisticRegression(max_iter=1000)

visual_model.fit(X_2d, y)

# =========================
# CREATE DECISION BOUNDARY GRID
# =========================

x_min, x_max = X_2d[:,0].min() - 1, X_2d[:,0].max() + 1
y_min, y_max = X_2d[:,1].min() - 1, X_2d[:,1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# =========================
# PREDICT OVER GRID
# =========================

Z = visual_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# =========================
# PLOT
# =========================

plt.figure(figsize=(14,10))

# decision regions
plt.contourf(xx, yy, Z, alpha=0.3)

# =========================
# PLOT DOCUMENT POINTS
# =========================

for i in range(len(X_2d)):

    x = X_2d[i][0]
    y_val = X_2d[i][1]

    # HAM
    if y.iloc[i] == 0:
        plt.scatter(
            x,
            y_val,
            marker='o',
            label='HAM' if i == 0 else ""
        )

    # SPAM
    elif y.iloc[i] == 1:
        plt.scatter(
            x,
            y_val,
            marker='^',
            label='SPAM' if i == 1 else ""
        )

    # SMISHING
    else:
        plt.scatter(
            x,
            y_val,
            marker='s',
            label='SMISHING' if i == 2 else ""
        )

# =========================
# LABELS
# =========================

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.title("TF-IDF + Logistic Regression Visualization")

plt.legend()

plt.show()