import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./Housing.csv")
corr = df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap='coolwarm')
sns.scatterplot(x=df["area"],y=df["price"])
plt.show()