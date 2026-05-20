import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -----------------------
# DATA
# -----------------------
data = load_breast_cancer()


X = data.data
y = data.target

x_train,x_test ,y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# -----------------------
# INIT PARAMETERS
# -----------------------
w = np.zeros(x_train.shape[1])
b = 0

lr = 0.1
epoch = 200

loss_history = []
z_history = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------
# TRAINING LOOP
# -----------------------
for i in range(epoch):
    z = np.dot(x_train, w) + b
    y_pred = sigmoid(z)

    dw = (1 / len(x_train)) * np.dot(x_train.T, (y_pred - y_train))
    db = (1 / len(x_train)) * np.sum(y_pred - y_train)

    w -= lr * dw
    b -= lr * db

    loss = -np.mean(y_train*np.log(y_pred + 1e-9) + (1-y_train)*np.log(1-y_pred + 1e-9))

    loss_history.append(loss)
    z_history.append(z.copy())

    if i % 50 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

# -----------------------
# LOSS CURVE PLOT
# -----------------------
plt.figure(figsize=(10, 4))
plt.plot(loss_history)
plt.title("Loss Curve (Binary Cross Entropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# testing model 

z_test = np.dot(x_test,w)+b
y_pred_test = sigmoid(z_test)

y_pred_labels = (y_pred_test >= 0.5).astype(int)


accuracy = np.mean(y_pred_labels == y_test)

print(f"accuracy: {accuracy}")

indices = np.arange(len(y_test))

plt.figure(figsize=(10,5))

# Class 0
plt.scatter(indices[y_test == 0], 
            y_pred_test[y_test == 0], 
            color='blue', alpha=0.5, label='Actual Class 0')

# Class 1
plt.scatter(indices[y_test == 1], 
            y_pred_test[y_test == 1], 
            color='orange', alpha=0.5, label='Actual Class 1')

# Wrong predictions
wrong = (y_pred_labels != y_test)
plt.scatter(indices[wrong], 
            y_pred_test[wrong], 
            color='red', s=80, label='Wrong')

plt.axhline(0.5, color='black', linestyle='--')

plt.title("Predictions with Errors Highlighted")
plt.xlabel("Sample Index")
plt.ylabel("Predicted Probability")
plt.legend()
plt.show()