import pandas as pd
from sklearn.model_selection import train_test_split

column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
                'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
                'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
df = pd.read_csv('wine.data', names=column_names)
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model_ham = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model_ham.fit(X_train, y_train)

y_pred = model_ham.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Ham Veri Accuracy (Doğruluk): %{acc*100:.2f}")
plt.figure(figsize=(8,5))
plt.plot(model_ham.loss_curve_)
plt.title("Ham Veri - Loss (Kayıp) Eğrisi")
plt.xlabel("Iterasyon (Epoch)")
plt.ylabel("Loss Değeri")
plt.grid(True)
plt.show()