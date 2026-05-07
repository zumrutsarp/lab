import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_excel("CarSales.xlsx")
df_clean = df.dropna(subset=['Horsepower', 'Price_in_thousands'])

X_simple = df_clean [['Horsepower']]
y_simple = df_clean ['Price_in_thousands']

model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)

y_pred_simple = model_simple.predict(X_simple)
mse_simple = mean_squared_error(y_simple, y_pred_simple)
r2_simple = r2_score(y_simple, y_pred_simple) 
print("Simple Linear Regression")
print("Mean Squared Error:", mse_simple)
print("R-squared:", r2_simple)     
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y_simple, color='steelblue', label='Data Points')    

#visualization
plt.figure(figsize=(10, 6))
# Gerçek veri noktalarını dağılım grafiği olarak çizdiriyoruz
plt.scatter(X_simple.values, y_simple.values, color='steelblue', label='Data Points', alpha=0.6)
plt.plot(X_simple.values, y_pred_simple, color='red', linewidth=2, label='Simple Linear Regression Fit')

plt.title('Simple Linear Regression: Horsepower vs Price_in_thousands')
plt.xlabel('Horsepower')
plt.ylabel('Price (in $1000s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()  
