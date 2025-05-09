import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import ydf

price_history = pd.read_csv('data/ETH_USD_11_20_2020-1_19_2021_historical_prices.csv', sep=';')
whales = pd.read_csv('data/transactions_aggregated_21_22.csv')
validators = pd.read_csv('data/validators_aggregated_21_22.csv')

current_data = pd.read_csv('data/transaction_aggregated_25.csv')
current_data = pd.read_csv('data/validators_aggregated_25.csv')


## Data Engineering for whales and validators
def main_data_engineering(num_labels = 3,
                          validators = None,
                          whales = None,
                          do_test_train_split = True,
                          window_length = 14): # -> X_train, X_test, y_train, y_test
    if not validators or not whales:
        raise ValueError("Validators and whales dataframes cannot be empty.")
    
    # Basic cleaning
    whales['date'] = pd.to_datetime(whales['datetime']).dt.date
    validators['date'] = pd.to_datetime(validators['datetime']).dt.date

    ## Price history in price deltas includes datetime, price delta, and eth_volume
    price_deltas = price_history.copy()[['timeOpen', 'close', 'volume']]
    # price_deltas['delta'] = price_deltas['close'].diff().shift(-1)
    price_deltas['delta'] = price_deltas['close'].pct_change()                  # Percent change function here
    price_deltas['date'] = pd.to_datetime(price_deltas['timeOpen']).dt.date


    # Adjustable number of labels
    labels = [x for x in range(num_labels)]

    percentile_intervals = np.linspace(0, 100, num_labels + 1)
    percentiles = np.percentile(price_deltas['delta'].dropna(), percentile_intervals[1:-1])
    
    print(f"Percentiles ({percentile_intervals[1:-1]}):", percentiles)

    levels = [-np.inf] + list(percentiles) + [np.inf]
    price_deltas['labels'] = pd.cut(x=price_deltas['delta'], bins=levels, labels=labels)

    # Aggregate all against date
    whale_aggregated = whales.groupby('date').agg(
        whale_avg_valueEth=('valueETH', 'mean'),
        whale_var_valueEth=('valueETH', 'var'),
        whale_avg_gasPrice=('gasPrice', 'mean')
    )

    validators_aggregated = validators.groupby('date').agg(
        validator_count=('blockHash', 'nunique'),
        validator_gas_price=('gasPrice', 'mean'),
    )


    # Dataset merging onto labels
    _intermediate = pd.merge(
        price_deltas,
        whale_aggregated,
        on='date',
        how='left'
    )

    labeled_data = pd.merge(
        _intermediate,
        validators_aggregated,
        on='date',
        how='left'
    )

    # Sort inplace
    labeled_data.sort_values(by='date', inplace=True, ascending=True) 
    labeled_data.dropna(inplace=True)

    print(f"Merge results: Initial obs: {len(whale_aggregated)}, Final obs {len(labeled_data)}")

    y = labeled_data['labels'].astype(int)
    X = labeled_data[['whale_avg_valueEth', 'whale_var_valueEth', 'whale_avg_gasPrice', 'validator_count', 'validator_gas_price']]
    

    # Build windows
    def create_windows(data, window_size):
        windows = []
        for i in range(len(data) - window_size + 1):
            windows.append(data[i:i + window_size])
        return np.array(windows)
    
    # Create windowed sequences for training and testing
    X_windowed = create_windows(X.values, window_length)
    y_windowed = y[window_length-1:]


    # Split into train and test optionally
    if do_test_train_split:
        return train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        return X, None, y, None



## Running Data Engineering for various model input format needs

## For YDF format:
Xt, Xe, yt, ye = main_data_engineering(num_labels=3, validators=validators, whales=whales, do_test_train_split=True)

# Train
ydf_train = pd.DataFrame(
    data=Xt.reshape(Xt.shape[0], -1),
    columns=[f"feature_{i}" for i in range(Xt.shape[1] * Xt.shape[2])]
)
ydf_train['labels'] = yt.values

# Test
ydf_test = pd.DataFrame(
    data=Xe.reshape(Xe.shape[0], -1),
    columns=[f"feature_{i}" for i in range(Xe.shape[1] * Xe.shape[2])]
)
ydf_test['labels'] = ye.values

model = ydf.RandomForestLearner(
    label= "labels",
    task=ydf.Task.CLASSIFICATION,
    num_trees=1000, # Default 300
    max_depth=8, #Default 16
    min_examples=10, #Default 5
).train(ydf_train,)

# Evaluate the model
evaluation_train = model.evaluate(ydf_train)
evaluation_test = model.evaluate(ydf_test)

y_pred_test = model.predict(ydf_test)

print(f"Train Accuracy: {evaluation_train.accuracy}")
print(f"Test Accuracy: {evaluation_test.accuracy}")
print(f"Train Confusion Matrix")
print(evaluation_train.confusion_matrix)
print(f"Test Confusion Matrix")
print(evaluation_test.confusion_matrix)


# model = LogisticRegression(max_iter=1000, random_state=42) ## 40% Test
# model = MLPClassifier([24, 24, 24], max_iter=10000, random_state=42, solver='adam', activation='logistic') ## 50% Test

# Fit the model on the training data
# model.fit(X_train, y_train)

# Predict on training and test data
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)

# Calculate accuracy - FIXED to use windowed labels
# train_accuracy = accuracy_score(y_train_windowed, y_train_pred)
# test_accuracy = accuracy_score(y_test_windowed, y_test_pred)

# print(f"Train Accuracy: {train_accuracy}")
# print(f"Test Accuracy: {test_accuracy}")

# # Generate the confusion matrix - FIXED to use windowed labels
# conf_matrix = confusion_matrix(y_test_windowed, y_test_pred, labels=range(len(labels)))

# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix Heatmap')
# plt.show()


def calculate_model_returns(predictions, price_deltas, num_labels,
                             invest_on_tie=True,  plot_results=True, investment_rate = 1.0):
    """
    Calculate expected investment returns based on model predictions and historical price changes.
    
    Parameters:
    -----------
    predictions : array-like
        Vector of model predictions (class labels).
    price_deltas : array-like
        Array of historical price changes (returns/deltas) corresponding to each prediction.
    num_labels : int
        Number of classification labels (e.g., 5 for labels 0-4).
    invest_on_tie : bool, optional, default=True
        If True, invest when the prediction is at the midpoint label (in case of ties).
    plot_results : bool, optional, default=True
        Whether to generate and display the performance plot.
    investment_rate : float, optional, default=1.0
        Proportion of capital to invest in each valid prediction.

    Returns:
    --------
    dict
        Dictionary containing the following keys:
        - 'total_return_pct': Total percentage return of the model's investment strategy.
        - 'benchmark_return_pct': Total percentage return of a buy-and-hold benchmark strategy.

    Notes:
    ------
    - The function simulates an investment strategy based on the model's predictions and compares it to a benchmark.
    - The benchmark assumes a buy-and-hold strategy with no prediction-based adjustments.
    - The function calculates and plots the capital history for both the model and the benchmark.
    """
    
    model_history = [1] # start with 1 unit of capital
    benchmark_history = [1] # Floats of unit change (-1% change is -.01)
    investing_schedule = [] # 0 = not investing, 1 = investing

    # check if investing
    def check_investment(x) -> bool:
        if num_labels <2:
            if x == 0 and invest_on_tie:
                return True
            else:
                return x > 0
        elif num_labels % 2 == 0:
            return x > num_labels // 2
        else:
            return x >= num_labels // 2 if invest_on_tie else x > num_labels // 2
        
    for i in range(len(predictions)):
        # Update benchmark history regardless of prediciton
        benchmark_history.append(model_history[-1] * (price_deltas[i] + 1))
        
        # Check if the prediction is valid for investment
        if not check_investment(predictions[i]):
            model_history.append(model_history[-1])
            
            investing_schedule.append(0)
            continue

        else:
        # Calculate the capital change based on the return value
            capital_change = model_history[-1] * price_deltas[i] * investment_rate
            new_capital = model_history[-1] + capital_change
            
            # Add to histories
            model_history.append(new_capital)
            investing_schedule.append(1)

    total_return_pct = (model_history[-1] - model_history[0]) / model_history[0] * 100
    benchmark_return_pct = (benchmark_history[-1] - benchmark_history[0]) / benchmark_history[0] * 100

    if plot_results:
        plt.figure(figsize=(12, 6))
        
        # Plot capital history
        plt.plot(model_history, label='Model', color='blue')
        plt.plot(benchmark_history, label='Benchmark', color='red')
        
        # Highlight regions where investing occurred
        for i in range(len(investing_schedule)):
            if investing_schedule[i] == 1:
                plt.axvspan(i, i + 1, color='green', alpha=0.3, label='Investing' if i == 0 else "")
        
        plt.title(f'Investment Simulation Results\n'
                  f'Model Return: {total_return_pct:.2f}%, Buy & Hold: {benchmark_return_pct:.2f}%')
        plt.xlabel('Days')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    # Return simulation results
    results = {
        'total_return_pct': total_return_pct,
        'benchmark_return_pct': benchmark_return_pct,
    } 
    
    return results


# Get predictions from the model
predictions = np.argmax(model.predict(ydf_test), axis=1)
print(predictions)

# Extract the 'delta' column from price_deltas for the corresponding test data
price_deltas_test = price_deltas['delta'].iloc[-len(y_test_windowed):].to_numpy()

# Calculate model returns
calculate_model_returns(predictions=predictions, price_deltas=price_deltas_test, num_labels=num_labels,invest_on_tie=False, investment_rate=0.8)


