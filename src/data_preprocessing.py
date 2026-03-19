import pandas as pd
import joblib

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.drop("Id", axis=1)

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]

   
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    return X, y