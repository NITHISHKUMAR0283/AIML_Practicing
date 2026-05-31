import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
class SVM:
    def __init__(self,learning_rate = 0.1,c = 0.5,n_iters=100):
        self.lr = learning_rate
        self.c = c
        self.n_iter = n_iters
        self.w = None
        self.b = None
        
    def fit(self,X,y):
        x_row, x_cols =X.shape
        y_ = np.where(y<0,-1,1)

        self.w = np.zeros(x_cols)
        self.b=0
        for _ in range(self.n_iter):
            for idx,x_val in enumerate(X):
                condition = y_[idx]*(np.dot(x_val,self.w)+self.b)>=1

                if condition:
                    self.w = self.w - self.lr * self.w
                else:
                    self.w = self.w - self.lr * (self.w - self.c * y_[idx] * x_val)
                    self.b = self.b + self.lr * self.c * y_[idx]
    def predict(self,X):
        value = np.dot(X,self.w)+self.b
        return np.sign(value)
data = load_breast_cancer()
X = data.data
y = data.target

y = np.where(y==0,-1,1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

pca = PCA(n_components=2)

X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

custom_svm = SVM(learning_rate=0.00001, c=200.0 ,n_iters=10000)
custom_svm.fit(X_train_2d, y_train)

sk_svm = LinearSVC()
sk_svm.fit(X_train_2d, y_train)

y_pred_custom = custom_svm.predict(X_test_2d)
y_pred_sklearn = sk_svm.predict(X_test_2d)

print("Custom SVM Accuracy:", accuracy_score(y_test, y_pred_custom))
print("Sklearn SVM Accuracy:", accuracy_score(y_test, y_pred_sklearn))

def plot_boundary(ax, model, X, y, title, accuracy):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    
    ax.contourf(xx, yy, Z, alpha=0.25)

    
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=60,              
        edgecolors='black',
        linewidths=1.2,
        alpha=0.9
    )

    ax.set_title(f"{title}\nAccuracy = {accuracy:}")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
acc_custom = accuracy_score(y_test, y_pred_custom)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_boundary(
    axes[0],
    custom_svm,
    X_test_2d,
    y_test,
    "Custom SVM",
    acc_custom
)

plot_boundary(
    axes[1],
    sk_svm,
    X_test_2d,
    y_test,
    "LinearSVC",
    acc_sklearn
)

plt.tight_layout()
plt.show()