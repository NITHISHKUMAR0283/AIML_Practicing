from decisionTree import build_Tree
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()

X= data.data
y = data.target

X_train , X_test, y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)


def random_forest(X,y,n_estimators = 100):
    Forest = []  
    for i in range(n_estimators):
        ran_row_ind = np.random.choice(X.shape[0],size = X.shape[0],replace=True)
        X_rand = X[ran_row_ind]
        y_rand = y[ran_row_ind]
        tree = build_Tree(X_rand,y_rand,1,4)
        Forest.append(tree)
    return Forest
def final_prediction(Forest , X_test):
    value = []
    for tree in Forest:
        value.append(predict(tree,X_test))
        
    classes, count = np.unique(value,return_counts=True)
    max_ind = np.argmax(count)
    return classes[max_ind]

def predict(curr_node,X_test):
    if(curr_node.value !=None):
        return curr_node.value
    feature = curr_node.feature
    if(X_test[feature]<=curr_node.threshold):
        return predict(curr_node.left,X_test)
    else:
        return predict(curr_node.right,X_test)

Forest = random_forest(X_train,y_train,100)

prediction = []
for row in X_test:
    prediction.append(final_prediction(Forest,row))

accuracy = accuracy_score(y_test,prediction)

print(accuracy)