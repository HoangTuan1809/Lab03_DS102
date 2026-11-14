import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Linear_Regression import LinearRegression

df = pd.read_csv('Forest_Fires/forestfires.csv')
df_encoded = pd.get_dummies(df, columns=['day', 'month'], drop_first=True, dtype=np.float64)

X = df_encoded.drop(columns=['area'])
y = df_encoded['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean_train = np.mean(X_train.values, axis=0)
std_train = np.std(X_train.values, axis=0)
std_train[std_train == 0] = 1

X_train_scaled = (X_train.values - mean_train) / std_train
X_test_scaled = (X_test.values - mean_train) / std_train

model = LinearRegression()
model.fit(X_train_scaled, y_train)

def print_metrics(metrics):
    print(f'MSE: {metrics['mse']:,.2f}')
    print(f'RMSE: {metrics["rmse"]:.2f}')
    print(f'R2: {metrics["r2"]:.2f}')

print('Kết quả trên tập train')
print_metrics(model.evaluate(X_train_scaled, y_train))
print('Kết quả trên tập test')
print_metrics(model.evaluate(X_test_scaled, y_test))