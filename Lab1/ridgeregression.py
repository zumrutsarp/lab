import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Veriyi yükleme
df = pd.read_excel("CarSales.xlsx")

# --- D. RIDGE REGRESYON ---
# Boş verileri temizleme
df_d = df[['Horsepower', 'Price_in_thousands']].dropna()
X_d = df_d[['Horsepower']]
y_d = df_d['Price_in_thousands']

# Polinomiyal Dönüşüm (Yönergede degree=3 isteniyor)
poly = PolynomialFeatures(degree=3)
X_poly_d = poly.fit_transform(X_d)

# Ridge Modeli (Yönergeye göre alpha test ediyoruz)
alpha_val = 10
model_d = Ridge(alpha=alpha_val).fit(X_poly_d, y_d)

# Tahmin ve Metrikler
y_pred_d = model_d.predict(X_poly_d)   
mse_d = mean_squared_error(y_d, y_pred_d)
r2_d = r2_score(y_d, y_pred_d)

print("--- Ridge Regression Sonuçları ---")
print("Mean Squared Error:", mse_d)
print("R-squared:", r2_d)

# Görselleştirme (Grafik D)
plt.figure(figsize=(10, 6))
plt.scatter(X_d.values, y_d.values, color='steelblue', alpha=0.5, label='Veri Noktaları')

# Eğrinin pürüzsüz görünmesi için X aralığı oluşturma
X_range = np.linspace(X_d.min(), X_d.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model_d.predict(X_range_poly)

plt.plot(X_range, y_range_pred, color='orange', linewidth=3, label=f'Ridge (Alpha={alpha_val}) Fit')

plt.title('D - Ridge Regression (Effect of Alpha)')
plt.xlabel('Horsepower (HP)')
plt.ylabel('Price (in $1000s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()