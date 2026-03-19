import joblib
import pandas as pd

def predict_from_user_input():
    # Load model and training columns
    model = joblib.load("models/random_forest_model.pkl")
    columns = joblib.load("models/columns.pkl")

    print("\nEnter house details")

    area = float(input("Area: "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    floors = int(input("Floors: "))
    year = int(input("Year Built: "))
    location = input("Location (Downtown/Suburban/Rural): ")
    condition = input("Condition (Excellent/Good/Fair): ")
    garage = input("Garage (Yes/No): ")

    # Create input dataframe
    user_df = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Floors": [floors],
        "YearBuilt": [year],
        "Location": [location],
        "Condition": [condition],
        "Garage": [garage]
    })

    # One-hot encode user input
    user_df = pd.get_dummies(user_df)

    # Align with training columns
    user_df = user_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(user_df)

    print("\nPredicted House Price: ₹", round(prediction[0], 2))