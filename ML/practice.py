# import math
# import numpy as np
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

# import pandas as pd

# df = pd.read_csv("./Titanic-Dataset.csv")
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



