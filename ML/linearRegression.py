import numpy as np 
import matplotlib.pyplot as plt

from matplotlib import animation

X = np.random.randint(1, 10, size=5)     
Y = np.random.randint(-100, 0, size=5)
X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()
w = np.random.random()
b = np.random.random()

epoch = 1000
lr = 0.01
n = len(X)

w_history = []
b_history = []
loss_history = []


for i in range(epoch):
    
    y_pred = w*X + b
    dw = (-2/n) * np.sum(X*(Y-y_pred))
    db = (-2/n) * np.sum((Y - y_pred))
    b = b - lr * db
    w = w-(lr*dw)
    
    y_pred = w*X + b
    loss = (1/n) * np.sum((Y - y_pred)**2)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss}")

    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss)

    
w_vals = np.linspace(-2, 2, 50)
b_vals = np.linspace(-2, 2, 50)

W, B = np.meshgrid(w_vals, b_vals)
Z = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        y_pred = W[i,j]*X + B[i,j]
        Z[i,j] = (1/n) * np.sum((Y - y_pred)**2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface
ax.plot_surface(W, B, Z, alpha=0.6)

# Path (initial)
point, = ax.plot([], [], [], 'ro-', markersize=4)

ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("Loss")
ax.set_title("Gradient Descent on Loss Surface")

# Animation function
def update(frame):
    point.set_data(w_history[:frame], b_history[:frame])
    point.set_3d_properties(loss_history[:frame])
    return point,

ani = animation.FuncAnimation(
    fig, update, frames=len(w_history), interval=100, blit=False
)

plt.show()


plt.figure()
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss decreasing over time")
plt.show()


fig2, ax2 = plt.subplots()

ax2.scatter(X, Y)
line, = ax2.plot(X, w_history[0]*X + b_history[0])

ax2.set_title("Line Fitting Over Time")

def update_line(frame):
    y_pred = w_history[frame] * X + b_history[frame]
    line.set_ydata(y_pred)
    ax2.set_title(f"Iteration {frame}")
    return line,

ani2 = animation.FuncAnimation(
    fig2, update_line, frames=len(w_history), interval=50, blit=False
)

plt.show()