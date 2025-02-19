from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("./data/data_linearRegression.csv")

x = data["x"].values.reshape(-1, 1)
y = data["y"].values.reshape(-1, 1)



#建模
lr_model = LinearRegression()
lr_model.fit(x, y)

a = lr_model.coef_
b = lr_model.intercept_
print(a, b,flush=True)
#预测
predictions = lr_model.predict(x)
print(predictions)
#评估
mse = mean_squared_error(y, predictions) #越小越好
r2 = r2_score(y, predictions) #越接近1越好
print(mse, r2,flush=True)

plt.figure(figsize=(5,5))
plt.scatter(y,predictions)
plt.show()