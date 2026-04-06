# ================= INDUSTRY LEVEL HOUSE PRICE PROJECT =================

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ================= 1. DATA INGESTION =================
def load_data():
    df = pd.read_csv("Housing.csv")
    print("Data Loaded Successfully")
    return df


# ================= 2. PREPROCESSING =================
def preprocess(df):
    print("\nMissing Values:\n", df.isnull().sum())

    cat_cols = ['mainroad', 'guestroom', 'basement']

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df


# ================= 3. EDA WITH MATPLOTLIB =================
def perform_eda(df):
    print("\n--- EDA ---")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns)

    # Histogram
    df.hist(figsize=(10, 8))
    plt.show()

    # Price vs Area
    plt.figure()
    plt.scatter(df['area'], df['price'])
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Area vs Price")
    plt.show()

    # Bedrooms vs Price
    plt.figure()
    plt.scatter(df['bedrooms'], df['price'])
    plt.xlabel("Bedrooms")
    plt.ylabel("Price")
    plt.title("Bedrooms vs Price")
    plt.show()


# ================= 4. FEATURE ENGINEERING =================
def get_features(df):
    X = df[['area', 'bedrooms', 'bathrooms', 'stories',
            'mainroad', 'guestroom', 'basement', 'parking']]

    y = df['price']

    return X, y


# ================= 5. MODEL TRAINING =================
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# ================= 6. MODEL EVALUATION =================
def evaluate_models(models, X_test, y_test):
    results = {}

    print("\n--- Model Performance ---")

    for name, model in models.items():
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        results[name] = mse
        print(f"{name} MSE: {mse}")

    return results


# ================= 7. SELECT BEST MODEL =================
def select_best_model(models, results):
    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]

    print("\nBest Model:", best_model_name)

    pickle.dump(best_model, open("house_model.pkl", "wb"))
    print("Best model saved")

    return best_model


# ================= 8. PREDICTION =================
def predict():
    model = pickle.load(open("house_model.pkl", "rb"))

    print("\nEnter Details:")

    area = float(input("Area: "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    stories = int(input("Stories: "))
    mainroad = int(input("Mainroad (1 yes / 0 no): "))
    guestroom = int(input("Guestroom (1 yes / 0 no): "))
    basement = int(input("Basement (1 yes / 0 no): "))
    parking = int(input("Parking: "))

    data = [[area, bedrooms, bathrooms, stories,
             mainroad, guestroom, basement, parking]]

    result = model.predict(data)[0]

    print("\nEstimated Price:", round(result, 2))


# ================= MAIN =================
def main():
    df = load_data()
    df = preprocess(df)

    perform_eda(df)

    X, y = get_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    select_best_model(models, results)

    predict()

    print("\nPROJECT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()