import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


## Data Engineering for whales and validators
def feature_engineering(num_labels = 3,
                          validators = None,
                          whales = None,
                          price_history = None, 
                          do_test_train_split = True,
                          window_length = 14): # -> X_train, X_test, y_train, y_test
    """
    Feature engineering from raw transactions and historical data into
    optionally split dataframes. Exports all data as either a test-train split,
    or as X, y dataframes.

    Returns:
    --------
    Tuple:
        - X_train, X_test, y_train, y_test
    """
    if not validators or not whales or not price_history:
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
