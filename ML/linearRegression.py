import numpy as np
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import pandas as pd

df = pd.DataFrame({
    'area': [800, 1000, 1200, 1500, 1800, 2000],
    'bedrooms': [1, 2, 2, 3, 3, 4],
    'price': [2, 3, 3.5, 5, 6, 7]
})
X = df[['area','bedrooms']]
y = df['price']
model = LinearRegression()
model.fit(X,y)

y_predict = model.predict(X)
print(model.coef_)
print(model.intercept_)

# Create 3D plot
fig = pt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot actual data points
ax.scatter(X['area'], X['bedrooms'], y, color='red', s=100, label='Actual prices')

# Create mesh grid for the plane
area_range = np.linspace(X['area'].min(), X['area'].max(), 10)
bedrooms_range = np.linspace(X['bedrooms'].min(), X['bedrooms'].max(), 10)
area_mesh, bedrooms_mesh = np.meshgrid(area_range, bedrooms_range)

# Predict prices for mesh
mesh_points = np.c_[area_mesh.ravel(), bedrooms_mesh.ravel()]
price_mesh = model.predict(mesh_points).reshape(area_mesh.shape)

# Plot the regression plane
ax.plot_surface(area_mesh, bedrooms_mesh, price_mesh, alpha=0.5, cmap='viridis', label='Regression plane')

ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.set_title('Linear Regression Plane')
pt.show()

