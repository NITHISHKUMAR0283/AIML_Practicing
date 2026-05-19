import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def split(feature, X, y, threshold):
    left_index = X[:, feature] <= threshold
    right_index = X[:, feature] > threshold

    X_left = X[left_index]
    X_right = X[right_index]

    y_left = y[left_index]
    y_right = y[right_index]

    return X_left, X_right, y_left, y_right


def gini(y):
    labels, count = np.unique(y, return_counts=True)
    probs = count / len(y)
    return 1 - np.sum(probs ** 2)


def best_split(X, y):
    _, features = X.shape
    best_feature = None
    best_threshold = None
    total_sample = len(y)
    lowest_impurity = float('inf')

    for feature in range(features):
        possible_values = np.unique(X[:, feature])
        possible_values = np.sort(possible_values)

        for i in range(len(possible_values) - 1):
            threshold = (possible_values[i] + possible_values[i + 1]) / 2

            X_left, X_right, y_left, y_right = split(feature, X, y, threshold)

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            impurity_left = gini(y_left)
            impurity_right = gini(y_right)

            total_impurity = (
                impurity_left * (len(y_left) / total_sample) +
                impurity_right * (len(y_right) / total_sample)
            )

            if total_impurity < lowest_impurity:
                lowest_impurity = total_impurity
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


class Node:
    def __init__(self,
                 feature=None,
                 threshold=None,
                 left=None,
                 right=None,
                 value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def construct_tree(X, y, depth, limit):
    if len(np.unique(y)) == 1 or depth == limit:
        value = np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]
        return Node(value=value)

    feature, threshold = best_split(X, y)

    if feature is None:
        value = np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]
        return Node(value=value)

    X_left, X_right, y_left, y_right = split(feature, X, y, threshold)

    if len(y_left) == 0 or len(y_right) == 0:
        value = np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]
        return Node(value=value)

    head = Node()
    head.feature = feature
    head.threshold = threshold

    head.left = construct_tree(X_left, y_left, depth + 1, limit)
    head.right = construct_tree(X_right, y_right, depth + 1, limit)

    return head


def visualize_tree(node, feature_names, dot=None, parent=None, edge_label="root"):

    if dot is None:
        dot = Digraph()

    node_id = str(id(node))

    if node.value is not None:
        dot.node(node_id, f"Leaf: {node.value}")
    else:
        feature_name = feature_names[node.feature]
        dot.node(node_id, f"{feature_name} <= {node.threshold}")

    if parent is not None:
        dot.edge(parent, node_id, label=edge_label)

    if node.left:
        visualize_tree(node.left, feature_names, dot, node_id, "True")

    if node.right:
        visualize_tree(node.right, feature_names, dot, node_id, "False")

    return dot


def predict(X, head):
    if head.value is not None:
        return head.value

    if X[head.feature] <= head.threshold:
        return predict(X, head.left)
    else:
        return predict(X, head.right)



data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = construct_tree(X_train, y_train, depth=0, limit=4)

buildin_model = DecisionTreeClassifier(criterion='gini',max_depth=4)
buildin_model.fit(X_train,y_train)
buildin_pred = buildin_model.predict(X_test)
accuracy_building = accuracy_score(y_test,buildin_pred)
print("accuracy of building sklearn model ",accuracy_building)

predictions = []
for i in range(len(X_test)):
    predictions.append(predict(X_test[i], model))

predictions = np.array(predictions)

accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)

dot = visualize_tree(model, feature_names)
dot.render("tree", format="png", view=True)

plt.figure(figsize=(20,10))
plot_tree(buildin_model,filled=True,
          feature_names=feature_names,
          class_names=data.target_names,
          )
plt.show()