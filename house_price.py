import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("Housing.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "").str.replace("_", "")

# Encode
df['mainroad'] = df[df.columns[df.columns.str.contains("mainroad")][0]].map({'yes':1,'no':0})
df['guestroom'] = df[df.columns[df.columns.str.contains("guestroom")][0]].map({'yes':1,'no':0})
df['basement'] = df[df.columns[df.columns.str.contains("basement")][0]].map({'yes':1,'no':0})

# Features
X = df[['area','bedrooms','bathrooms','stories',
        'mainroad','guestroom','basement','parking']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
lr = LinearRegression()
rf = RandomForestRegressor()
xgb = XGBRegressor()

# Train
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predict
lr_mse = mean_squared_error(y_test, lr.predict(X_test))
rf_mse = mean_squared_error(y_test, rf.predict(X_test))
xgb_mse = mean_squared_error(y_test, xgb.predict(X_test))

print("LR:", lr_mse)
print("RF:", rf_mse)
print("XGB:", xgb_mse)

# Best model
best_model = min(
    [(lr_mse, lr), (rf_mse, rf), (xgb_mse, xgb)],
    key=lambda x: x[0]
)[1]

# Save model
pickle.dump(best_model, open("house_model.pkl", "wb"))

print("🚀 Best model saved!")