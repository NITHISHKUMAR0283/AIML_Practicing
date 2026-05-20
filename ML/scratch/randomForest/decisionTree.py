import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
class Tree:
    def __init__(self,left = None ,right = None , feature = None , threshold = None , value = None  ):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
def split(X,y,feature,threshold):
    x_left = X[:,feature]<=threshold
    x_rigth = X[:,feature]>threshold
    return X[x_left],X[x_rigth],y[x_left],y[x_rigth]
def gini(y):
    if(len(y)==0 or len(y)==1): return 0
    classes,counts = np.unique(y,return_counts= True)
    impurity = 1
    for i  in range(len(counts)):
        impurity -= ((counts[i])/len(y))**2
    return impurity

def best_split(X,y):
    rows,features = X.shape
    best_feature = None
    best_threshold = None
    best_impurity = 2
    count = features/3 if features/3<=0 else 1 
    rand_feature = np.random.choice(features,int(count),replace=False)
    
    for feature in rand_feature:
        column = np.sort(X[:,feature])   

        for i in range(len(column)-1):
            average = (column[i]+column[i+1])/2
            x_left , x_right , y_left,y_right = split(X,y,feature,average)
            if(len(y_left)==0 or len(y_right)==0):
                continue
            l_gini , r_gini = gini(y_left),gini(y_right)
            impurity = l_gini*(len(y_left)/len(y))+ r_gini*(len(y_right)/len(y))
            if impurity <best_impurity:
                best_impurity =impurity
                best_feature = feature
                best_threshold = average
    return best_feature,best_threshold,best_impurity

def build_Tree(X,y,depth,limit):
    current = Tree()
    if(len(np.unique(y))==1 or depth == limit):
        classes,count = np.unique(y,return_counts=True)
        index = count.argmax()
        return Tree(value=classes[index])
    best_feature,best_threshold,best_impurity = best_split(X,y)
    
    x_left,x_right , y_left,y_right = split(X,y,best_feature,best_threshold)
    current.threshold = best_threshold
    current.feature = best_feature
    current.left, current.right=build_Tree(x_left,y_left,depth+1,limit),build_Tree(x_right,y_right,depth+1,limit)
    return current


