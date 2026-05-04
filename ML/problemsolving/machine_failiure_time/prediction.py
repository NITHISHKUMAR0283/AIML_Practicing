import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_excel("./ENB2012_data.xlsx")


X = df.drop(["Y1","Y2"],axis=1)

y = df[["Y1","Y2"]]

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)



cols = df.select_dtypes(include=['float64','int64']).columns.tolist()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')

print(f"Total Combined Accuracy: {r2_score(y_test, y_pred)*100:.2f}%")
print(f"Heating Load (Y1) Accuracy: {r2_scores[0]*100:.2f}%")
print(f"Cooling Load (Y2) Accuracy: {r2_scores[1]*100:.2f}%")

    

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test.iloc[:, 0], y_pred[:, 0], alpha=0.5, color='orange')
plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 'k--')
plt.title("Heating Load: Actual vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test.iloc[:, 1], y_pred[:, 1], alpha=0.5, color='blue')
plt.plot([y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], 'k--')
plt.title("Cooling Load: Actual vs Predicted")

plt.tight_layout()
plt.show()
