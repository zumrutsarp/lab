import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("CarSales.xlsx") # 


# --- C. POLİNOMİYAL REGRESYON (Degree=3) ---
poly = PolynomialFeatures(degree=3) # [cite: 12]
X_poly = poly.fit_transform(X_a)
model_c = LinearRegression().fit(X_poly, y_a)
y_pred_c = model_c.predict(X_poly)


# SONUÇLARI YAZDIRMA (Tabloyu buradan doldurabilirsin) 
results = {
    
    "Polynomial (Deg:3)": (y_a, y_pred_c)
    
}

print("\n--- LABORATUVAR SONUÇLARI ---")
for name, (y_true, y_pred) in results.items():
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MSE: {mse:.2f}, R-Squared: {r2:.4f}")

# GÖRSELLEŞTİRME: POLİNOMİYAL (C)
plt.figure(figsize=(10, 5))
plt.scatter(X_a.values, y_a.values, color='gray', alpha=0.5, label='Veri')
X_range = np.linspace(X_a.min(), X_a.max(), 100).reshape(-1, 1)
plt.plot(X_range, model_c.predict(poly.transform(X_range)), color='blue', label='Polynomial Fit')
plt.title("C - Polynomial Regression")
plt.legend()
plt.show()