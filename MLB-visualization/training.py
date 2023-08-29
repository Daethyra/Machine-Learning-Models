import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle

def load_data(csv_path: str):
    """Load the time series data from CSV."""
    data = pd.read_csv(csv_path)
    return data

def preprocess_data(data: pd.DataFrame, target_feature: str):
    """Preprocess the data by selecting the target feature and making it stationary if needed."""
    time_series_data = data[target_feature]
    return time_series_data

def train_model(time_series_data, order=(5,1,0)):
    """Train the ARIMA model with the given order."""
    model = ARIMA(time_series_data, order=order)
    model_fit = model.fit()
    return model_fit

def save_model(model_fit, filename='model.pkl'):
    """Save the trained model to a file."""
    with open(filename, 'wb') as file:
        pickle.dump(model_fit, file)

def train(csv_path: str, target_feature: str):
    """Main training function to load, preprocess, train, and save the model."""
    data = load_data(csv_path)
    time_series_data = preprocess_data(data, target_feature)
    model_fit = train_model(time_series_data)
    save_model(model_fit)

if __name__ == "__main__":
    csv_path = "data/data.csv" # Update with the correct path to your CSV file
    target_feature = "home_runs" # Update with the target feature you want to forecast
    train(csv_path, target_feature)
