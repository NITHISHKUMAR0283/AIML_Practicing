import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# -----------------------
# DATA
# -----------------------
data = load_breast_cancer()
X = data.data
y = data.target

# scale features
X = StandardScaler().fit_transform(X)

# -----------------------
# INIT PARAMETERS
# -----------------------
w = np.zeros(X.shape[1])
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
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)

    dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
    db = (1 / len(X)) * np.sum(y_pred - y)

    w -= lr * dw
    b -= lr * db

    loss = -np.mean(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))

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

# -----------------------
# Z-SPACE ANIMATION
# -----------------------
fig, ax = plt.subplots(figsize=(14, 6))

x_idx = np.arange(len(y))

# stable axis range
all_z = np.concatenate(z_history)
z_min, z_max = all_z.min(), all_z.max()

sc0 = ax.scatter([], [], color='red', label='Class 0')
sc1 = ax.scatter([], [], color='blue', label='Class 1')
line = ax.axhline(0, color='black')

ax.set_xlim(0, len(y))
ax.set_ylim(z_min - 2, z_max + 2)

ax.set_title("Z-space Evolution in Logistic Regression")
ax.set_xlabel("Sample Index")
ax.set_ylabel("z value")
ax.legend()

# -----------------------
# UPDATE FUNCTION
# -----------------------
def update(frame):
    z = z_history[frame]

    sc0.set_offsets(np.c_[x_idx[y == 0], z[y == 0]])
    sc1.set_offsets(np.c_[x_idx[y == 1], z[y == 1]])

    ax.set_title(f"Z-space at Epoch {frame}")

    return sc0, sc1, line

# -----------------------
# ANIMATION
# -----------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(z_history),
    interval=100,
    blit=False,
     repeat=False
)
plt.scatter(range(len(z)), z, c=y, cmap='bwr')
plt.axhline(0, color='black')
plt.title("z separated by class")
plt.xlabel("Sample index")
plt.ylabel("z")
plt.show()