import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


df = pd.read_csv("./Housing.csv")


binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_columns:
    df[col]=df[col].map({'yes':1,"no":0})


df= pd.get_dummies(df,columns=['furnishingstatus'],drop_first=True).astype(int)


q1  = df['area'].quantile(0.25)
q3  = df['area'].quantile(0.75)

iqr = q3-q1
lowerbound = q1-iqr*1.5
upperbound = q3+iqr*1.5

df=df[(df['area']>=lowerbound) & (df['area']<=upperbound)]
q1_p = df['price'].quantile(0.25)
q3_p = df['price'].quantile(0.75)
iqr_p = q3_p - q1_p
df = df[(df['price'] >= q1_p - 1.5*iqr_p) & (df['price'] <= q3_p + 1.5*iqr_p)]

X = df.drop('price',axis=1)
y = df['price']

x_train,x_test , y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(f"model accuracy {r2_score(y_test,y_pred)*100:.2f} %")



coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients.sort_values(by='Coefficient', ascending=False))

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.show()
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(scores.mean())