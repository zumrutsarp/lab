import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Veriyi Yükle ve Hazırla
df = pd.read_excel("CarSales.xlsx")

# Kullanılacak sütunlardaki boş (NaN) satırları temizle 
cols = ['Horsepower', 'Engine_size', 'Curb_weight', 'Price_in_thousands']
df_multi = df[cols].dropna()

X_multi = df_multi[['Horsepower', 'Engine_size', 'Curb_weight']] # Bağımsız değişkenler 
y_multi = df_multi['Price_in_thousands'] # Hedef değişken [cite: 2]

# 2. Modeli Eğit [cite: 4]
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# 3. Sonuçları Hesapla [cite: 17]
y_pred_multi = model_multi.predict(X_multi)
mse_multi = mean_squared_error(y_multi, y_pred_multi)
r2_multi = r2_score(y_multi, y_pred_multi)

print("--- Çoklu Doğrusal Regresyon Sonuçları ---")
print(f"MSE: {mse_multi:.2f}")
print(f"R-Squared: {r2_multi:.4f}")

# 4. 3 Boyutlu Görselleştirme (Grafik B) [cite: 10]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Gerçek veri noktaları
ax.scatter(df_multi['Horsepower'], df_multi['Engine_size'], df_multi['Price_in_thousands'], 
           color='steelblue', alpha=0.5, label='Veri Noktaları')

# Regresyon düzlemi için ızgara oluşturma
x_surf, y_surf = np.meshgrid(np.linspace(df_multi.Horsepower.min(), df_multi.Horsepower.max(), 10),
                             np.linspace(df_multi.Engine_size.min(), df_multi.Engine_size.max(), 10))

# Düzlem tahmini (Curb_weight'i sabit/ortalama kabul ediyoruz)
z_surf = (model_multi.coef_[0] * x_surf + 
          model_multi.coef_[1] * y_surf + 
          model_multi.coef_[2] * df_multi.Curb_weight.mean() + 
          model_multi.intercept_)

ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)

ax.set_xlabel('X: Horsepower') # [cite: 19]
ax.set_ylabel('Y: Engine size (L)') # [cite: 19]
ax.set_zlabel('Z: Price (in $1000s)') # [cite: 19]
plt.title('B - Multiple Linear Regression') # [cite: 10]
plt.show()