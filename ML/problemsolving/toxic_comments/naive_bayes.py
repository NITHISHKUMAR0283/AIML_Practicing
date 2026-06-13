import pandas as pd
from data_cleaning import preprocess 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df = pd.read_csv("./labeled_data.csv")

df = preprocess(df)
X=df["tweet"]
y = df['class']
print(df['class'].value_counts())

model = LinearSVC(class_weight='balanced',C=0.3)

x_train , x_test, y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

TfidfVector = TfidfVectorizer(
    analyzer='char',
    ngram_range=(1,10),
    max_features=20000
)
x_train = TfidfVector.fit_transform(x_train)
x_test = TfidfVector.transform(x_test)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test,y_pred))
sample = [
    "you are a bad guy i dont like you , i have anger on what you have done",
    "have a nice day",
    "go to hell idiot"
]

sample_df = pd.DataFrame({
    "tweet": sample
})

sample_df = preprocess(sample_df)

sample_vector = TfidfVector.transform(sample_df["tweet"])

predictions = model.predict(sample_vector)

label_map = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral"
}

for text, pred in zip(sample, predictions):

    print(f"\nText: {text}")

    print(f"Prediction: {label_map[pred]}")