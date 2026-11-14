import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from Linear_Regression import LinearRegression

def check_collinearity(X_df):
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(col, row, round(upper.loc[row, col], 4)) for col in upper.columns for row in upper.index if upper.loc[row, col] > 0.65]

    print('-----Các cặp có hệ số tương quan cao (|r| > 0.65)-----')
    if high_corr:
        for col1, col2, val in sorted(high_corr, key=lambda x: x[2], reverse=True):
            print(f' {col1} - {col2}: r = {val}')
    else:
        print('Không có cặp nào vượt ngưỡng')

    X_vif = sm.add_constant(X_df)
    vif_df = pd.DataFrame({
        'Feature': X_df.columns,
        'VIF': [variance_inflation_factor(X_vif.values, i+1) for i in range(X_df.shape[1])]
    }).sort_values(by='VIF', ascending=False).reset_index(drop=True)

    print('-----Bảng chỉ số VIF-----')
    print(vif_df.to_string(index=False))

def print_metrics(model, X_train, y_train, X_test, y_test, label):
    train_m = model.evaluate(X_train, y_train)
    test_m = model.evaluate(X_test, y_test)

    print(f'\n\t\t\t----- {label} (tập train) -----')
    print(f"MSE: {train_m['mse']} | RMSE: {train_m['rmse']} | R2: {train_m['r2']}")
    print(f'\t\t\t----- {label} (tập test) -----')
    print(f"MSE: {test_m['mse']} | RMSE: {test_m['rmse']} | R2: {test_m['r2']}")

df = pd.read_csv('Forest_Fires/forestfires.csv')
df_encoded = pd.get_dummies(df, columns=['day', 'month'], drop_first=True, dtype=np.float64)

X = df_encoded.drop(columns=['area'])
y = df_encoded['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chẩn đoán cộng tuyến trên nhóm biến số liên tục của tập Train
num_cols = df.drop(columns=['area']).select_dtypes(include='number').columns.tolist()
check_collinearity(X_train[num_cols])

# Loại bỏ cột cộng tuyến là DC và cột có phương sai bằng 0 nếu có
zero_var_cols = X_train.columns[X_train.std() == 0].tolist()
cols_dropped = list(set(['DC'] + zero_var_cols))

X_train_preprocessing = X_train.drop(columns=cols_dropped)
X_test_preprocessing = X_test.drop(columns=cols_dropped)

# Chuẩn hóa Z-Score
mean = np.mean(X_train_preprocessing.values, axis=0)
std = np.std(X_train_preprocessing.values, axis=0)
std[std == 0] = 1

X_train_prep_scaled = (X_train_preprocessing - mean) / std
X_test_prep_scaled = (X_test_preprocessing - mean) / std

# Bước so sánh original với preprocessed
model_original = LinearRegression()
model_original.fit(X_train, y_train)

model_prep = LinearRegression()
model_prep.fit(X_train_prep_scaled, y_train)

print_metrics(model_original, X_train, y_train, X_test, y_test, 'ORIGINAL')
print_metrics(model_prep, X_train_prep_scaled, y_train, X_test_prep_scaled, y_test, 'PREPROCESSED')