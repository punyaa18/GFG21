import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# =============================
# Setup
# =============================
df = pd.read_csv("Housing.csv")   # dataset in same folder
os.makedirs("outputs", exist_ok=True)

# Encode binary categorical features
def binary(x):
    return x.map({'yes':1,'no':0})

categorical = ['mainroad', 'guestroom', 'basement', 
               'hotwaterheating', 'airconditioning', 'prefarea']
df[categorical] = df[categorical].apply(binary)

# Drop furnishingstatus column
if "furnishingstatus" in df.columns:
    df = df.drop(columns=["furnishingstatus"])

# =============================
# 1. Skewness of SalePrice
# =============================
plt.figure(figsize=(8,5))
sns.histplot(df['price'], kde=True, color="blue")
plt.title(f"SalePrice Distribution (Skewness = {df['price'].skew():.2f})")
plt.savefig("outputs/saleprice_skew.png")
plt.close()

# =============================
# 2. Skewness of log-transformed SalePrice
# =============================
df['log_price'] = np.log1p(df['price'])

plt.figure(figsize=(8,5))
sns.histplot(df['log_price'], kde=True, color="green")
plt.title(f"Log(SalePrice) Distribution (Skewness = {df['log_price'].skew():.2f})")
plt.savefig("outputs/log_saleprice_skew.png")
plt.close()

# =============================
# 3. Correlation Matrix
# =============================
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("outputs/correlation_matrix.png")
plt.close()

# =============================
# 4. Model Evaluation
# =============================
X = df.drop(["price","log_price"], axis=1)
y = df["price"]

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "KNN Regressor": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append([name, r2, mae, rmse])

# Save evaluation as dataframe
eval_df = pd.DataFrame(results, columns=["Model","R²","MAE","RMSE"])
print(eval_df)
eval_df.to_csv("outputs/model_evaluation.csv", index=False)

# =============================
# 5. Model Performance Bar Plot
# =============================
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="R²", data=eval_df, palette="viridis")
plt.title("Model R² Comparison")
plt.savefig("outputs/model_r2.png")
plt.close()
