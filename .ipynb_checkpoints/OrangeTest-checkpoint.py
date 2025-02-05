import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ścieżka do folderu z danymi
data_path = r"E:\marcel.furs\Documents\Jane-Data"

# Wczytanie plików
train_file = os.path.join(data_path, "train.parquet")
lags_file = os.path.join(data_path, "lags.parquet")

# Wczytaj dane
train_data = pd.read_parquet(train_file)
lags_data = pd.read_parquet(lags_file)

# Wyświetl podstawowe informacje o danych
print(train_data.head())
print(train_data.info())

# Podstawowe statystyki
print(train_data.describe())

# Rozkład dla responder_6
plt.figure(figsize=(10, 6))
sns.histplot(train_data['responder_6'], bins=30, kde=True, color='blue')
plt.title("Rozkład responder_6")
plt.xlabel("Wartość responder_6")
plt.ylabel("Częstość")
plt.show()

# Korelacja cech z responder_6
correlation = train_data.corr()['responder_6'].sort_values(ascending=False)

# Wypisz 10 najbardziej skorelowanych cech
print("Najbardziej skorelowane cechy z responder_6:")
print(correlation.head(10))

# Heatmapa dla tych cech
top_features = correlation.index[:10]
plt.figure(figsize=(12, 8))
sns.heatmap(train_data[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelacja cech z responder_6")
plt.show()

# Eksport do CSV
output_file = os.path.join(data_path, "train_processed.csv")
train_data.to_csv(output_file, index=False)
print(f"Przetworzone dane zapisane do: {output_file}")
