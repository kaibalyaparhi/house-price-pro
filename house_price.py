import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ================= LOAD =================
# Force correct column names
columns = [
    'price','area','bedrooms','bathrooms','stories',
    'mainroad','guestroom','basement','hotwaterheating',
    'airconditioning','parking','prefarea','furnishingstatus'
]

df = pd.read_csv("Housing.csv", names=columns, header=None)

print("✅ Fixed Columns:", df.columns.tolist())

# ================= CLEAN =================
# Remove accidental header rows if repeated
df = df[df['price'] != 'price']

# Convert numeric columns
num_cols = ['price','area','bedrooms','bathrooms','stories','parking']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode yes/no columns
binary_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']

for col in binary_cols:
    df[col] = df[col].map({'yes':1,'no':0})

# Drop invalid rows
df = df.dropna()

# ================= FEATURES =================
X = df[['area','bedrooms','bathrooms','stories','parking']]
y = df['price']

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ================= MODELS =================
lr = LinearRegression()
rf = RandomForestRegressor()
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)

# ================= TRAIN =================
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ================= EVALUATE =================
lr_mse = mean_squared_error(y_test, lr.predict(X_test))
rf_mse = mean_squared_error(y_test, rf.predict(X_test))
xgb_mse = mean_squared_error(y_test, xgb.predict(X_test))

print("\n📊 Performance:")
print("LR:", lr_mse)
print("RF:", rf_mse)
print("XGB:", xgb_mse)

# ================= BEST MODEL =================
best_model = min(
    [(lr_mse, lr), (rf_mse, rf), (xgb_mse, xgb)],
    key=lambda x: x[0]
)[1]

# ================= SAVE =================
pickle.dump(best_model, open("house_model.pkl", "wb"))

# ================= FEATURE IMPORTANCE =================
try:
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": best_model.feature_importances_
        })
        fi.to_csv("feature_importance.csv", index=False)
        print("✅ Feature importance saved")
except:
    pass

print("🚀 Model saved successfully!")