from flask import Flask, render_template, request
from model import stock_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['ticker']
    try:
        # Get results for all models
        lstm_result = stock_prediction(stock_ticker, model_type='LSTM')
        rf_result = stock_prediction(stock_ticker, model_type='RandomForest')
        svm_result = stock_prediction(stock_ticker, model_type='SVM')
        lr_result = stock_prediction(stock_ticker, model_type='LinearRegression')

        # Pass all results to the template
        return render_template('analysis.html', 
                               ticker=stock_ticker, 
                               lstm_result=lstm_result, 
                               rf_result=rf_result, 
                               svm_result=svm_result, 
                               lr_result=lr_result)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
