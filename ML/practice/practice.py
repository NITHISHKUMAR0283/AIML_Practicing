import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./Titanic-Dataset.csv")
# data = [10, 20, 30, 40, 50]
# max = max(data)
# min = min(data)
# sumi = sum(data)
# mean = sumi/len(data)

# variance =(sum((x-mean)**2 for x in data))/len(data)
# std_deviation = math.sqrt(variance)

# print(max,min,sumi,mean,variance,std_deviation)

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# c=np.dot(a,b)
# print(a+b)
# print(c)

# x = np.array([1,2,3,4,5,6])
# w=3
# b=1


# y=w*x+b
# print(y)

# z = np.array([10])
# print(x+z)

# a = np.array([[1,2,3],
#               [4,5,6]])


# b = a.reshape(2,-1)
# print(b)
# print(a.T)

# print(np.random.randint(1,10))

# print(np.random.randint(1,1000,size=(2,7)))

# print(np.random.rand(2,3))
# print(np.random.normal(0,1,(5,5)))

# np.random.seed(42)
# print(np.random.rand())
# print(np.random.rand())


# print(df.describe())
# print(df.duplicated().sum())
# df["Age"].fillna(df["Age"].mean(),inplace=True)
# print(df.duplicated().sum())
# print(df.isnull().sum())

# survived = df[(df["Survived"]==0 )& (df["Sex"]=="female")]
# print(survived)

# sorted =df.sort_values(["Age","Fare"],ascending=[True,False])
# print(sorted)
# group = df.groupby("Pclass").size()
# print(group)


# df["AgeGroup"]=df["Age"].apply(lambda x: "Child" if x<18 else "Adult")
# print(df[(df["AgeGroup"]=="Child")])


# plt.plot([1,2,3,4],[5,6,7,8])
# df['Pclass'].value_counts().plot(kind='bar', color='orange')
# plt.title("Sample")
# plt.xlabel("samplex")
# plt.ylabel("sampley")
# plt.show()

# sns.countplot(data=df, x="Pclass",hue="Age")
# sns.scatterplot(x="Age",y="Fare",data=df,hue="Pclass",palette='viridis')
# corr = df.select_dtypes(include=['number']).corr()
# sns.heatmap(corr,annot=True,cmap='coolwarm')
# plt.show()
# sns.histplot(data=df,x="Age",bins=10)
# plt.show()
# sns.boxplot(data=df,x="Age")
# plt.show()
# sns.scatterplot(data=df,x="Age",y="Fare")
# plt.show()
# sns.pairplot(data=df,x_vars=df.columns,y_vars=df.columns)
# plt.show()

# Data Preprocessing

# print(df.info)
# print(df.describe())
# print(df.head())

# Q1= df["Age"].quantile(0.25)
# Q3= df["Age"].quantile(0.75)

# IQR = Q3-Q1

# min = Q1-1.5*IQR
# max = Q3+1.5*IQR

# outliers = (df[((df["Age"]>max) | (df["Age"]<min))])




# Removing missing values
df=df.dropna()
print(df.isnull().sum().sum())