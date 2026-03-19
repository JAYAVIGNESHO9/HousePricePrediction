from src.data_preprocessing import load_and_preprocess
from src.train_model import train_model
from src.user_predict import predict_from_user_input


def main():
    X, y = load_and_preprocess(
        "data/House Price Prediction Dataset.csv"
    )

    train_model(X, y)

    # Interactive prediction
    predict_from_user_input()


if __name__ == "__main__":
    main()