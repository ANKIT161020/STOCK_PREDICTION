import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def preprocess_data(stock_ticker, model_type='LSTM'):
    historical_days = 365
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=historical_days)).strftime('%Y-%m-%d')
    
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for the ticker '{stock_ticker}'.")

    data = data[['Close']]
    
    # Data Preprocessing
    if model_type == 'LSTM':
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        return data, data_scaled, scaler
    
    elif model_type in ['RandomForest', 'SVM', 'LinearRegression']:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data, data_scaled, scaler

def lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    return model

def random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def svm_model(X_train, y_train):
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    return model

def linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def get_predictions(model, X_train, y_train, model_type, scaler):
    if model_type == 'LSTM':
        predictions = model.predict(X_train)
        predictions = scaler.inverse_transform(predictions)
    else:
        predictions = model.predict(X_train)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_train, predictions)
    r2 = r2_score(y_train, predictions)
    
    return predictions, mse, r2

def stock_prediction(stock_ticker, model_type='LSTM'):
    data, data_scaled, scaler = preprocess_data(stock_ticker, model_type)

    X_train, y_train = [], []
    time_step = 60
    
    for i in range(time_step, len(data_scaled)):
        X_train.append(data_scaled[i-time_step:i, 0])
        y_train.append(data_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    if model_type == 'LSTM':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model = lstm_model(X_train, y_train)
    elif model_type == 'RandomForest':
        model = random_forest_model(X_train, y_train)
    elif model_type == 'SVM':
        model = svm_model(X_train, y_train)
    elif model_type == 'LinearRegression':
        model = linear_regression_model(X_train, y_train)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    
    predictions, mse, r2 = get_predictions(model, X_train, y_train, model_type, scaler)
    
    current_price = data['Close'].iloc[-1]
    predicted_final_price = predictions[-1][0]

    initial_investment = 10000
    shares_bought = initial_investment / current_price
    final_value = shares_bought * predicted_final_price
    return_percentage = ((final_value - initial_investment) / initial_investment) * 100

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], color='blue', label='Actual Stock Price')
    plt.plot(data.index[time_step:], predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{stock_ticker} Stock Price Prediction using {model_type}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Save the graph
    graph_folder = 'static/graphs'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
    graph_path = os.path.join(graph_folder, f'{stock_ticker}_prediction_{model_type}.png')
    plt.savefig(graph_path)
    plt.close()

    result = {
        'current_price': round(current_price, 2),
        'predicted_final_price': round(predicted_final_price, 2),
        'initial_investment': initial_investment,
        'final_value': round(final_value, 2),
        'return_percentage': round(return_percentage, 2),
        'mse': round(mse, 4),
        'r2': round(r2, 4),
        'graph_path': f'/static/graphs/{stock_ticker}_prediction_{model_type}.png'
    }
    return result
