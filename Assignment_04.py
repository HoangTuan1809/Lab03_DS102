import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Linear_Regression import LinearRegression as CustomLinearRegression

# 1. Nạp dữ liệu & tái tạo 2 bộ dữ liệu (Original và Preprocessed) giống Assignment 3
df = pd.read_csv('Forest_Fires/forestfires.csv')
df_encoded = pd.get_dummies(df, columns=['day', 'month'], drop_first=True, dtype=np.float64)

X = df_encoded.drop(columns=['area'])
y = df_encoded['area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bộ Preprocessed (Bỏ DC + Chuẩn hóa X + log1p(y))
zero_var_cols = X_train.columns[X_train.std() == 0].tolist()
cols_to_drop = list(set(['DC'] + zero_var_cols))

X_train_prep = X_train.drop(columns=cols_to_drop)
X_test_prep = X_test.drop(columns=cols_to_drop)

mean = np.mean(X_train_prep.values, axis=0)
std = np.std(X_train_prep.values, axis=0)
std[std == 0] = 1.0

X_train_prep_scaled = (X_train_prep.values - mean) / std
X_test_prep_scaled = (X_test_prep.values - mean) / std
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

def compare_models(X_train, y_train, X_test, y_test, data):
    # Custom NumPy Model
    custom_model = CustomLinearRegression()
    custom_model.fit(X_train, y_train)
    custom_metrics = custom_model.evaluate(X_test, y_test)

    # Scikit-Learn Model
    sk_model = SklearnLinearRegression()
    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    sk_mse = mean_squared_error(y_test, sk_pred)
    sk_rmse = np.sqrt(sk_mse)
    sk_r2 = r2_score(y_test, sk_pred)

    print(f"\n\t----- So sánh trên {data} (tập test) -----")
    print(f"Custom NumPy   MSE: {custom_metrics['mse']} | RMSE: {custom_metrics['rmse']} | R2: {custom_metrics['r2']}")
    print(f"Scikit-Learn   MSE: {sk_mse} | RMSE: {sk_rmse:} | R2: {sk_r2}")

# 2. Chạy đối chiếu trên cả 2 bộ dữ liệu
compare_models(X_train, y_train, X_test, y_test, "ORIGINAL DATA")
compare_models(X_train_prep_scaled, y_train_log, X_test_prep_scaled, y_test_log, "PREPROCESSED DATA")