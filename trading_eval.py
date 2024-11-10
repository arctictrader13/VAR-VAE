import numpy as np

def evaluate_trading_framework_component(samples, y_test, higher_percentile, lower_percentile, mean=False):
    """
    Evaluates a trading framework component based on specified percentiles or mean strategy.
    
    Args:
        samples (numpy.ndarray): The predicted samples for each time step.
        y_test (torch.Tensor or numpy.ndarray): The actual test values.
        higher_percentile (float): The upper percentile for prediction interval.
        lower_percentile (float): The lower percentile for prediction interval.
        mean (bool, optional): If True, calculates the mean forecast instead of percentiles. Defaults to False.
    
    Returns:
        dict: A dictionary containing evaluation metrics like overall correct percentage, profit, and Sharpe ratio.
    """
    # Calculate the specified percentiles and the mean across the samples for each time step
    percentile_higher = np.quantile(samples, higher_percentile / 100.0, axis=0)
    percentile_lower = np.quantile(samples, lower_percentile / 100.0, axis=0)
    
    # Flatten the actual test values
    actual_test_values = y_test.numpy() if hasattr(y_test, 'numpy') else y_test

    # Shift the actual test values to represent t+1 (excluding the last value)
    shifted_actual_values = actual_test_values[1:]
    shifted_percentile_higher = percentile_higher[:-1]
    shifted_percentile_lower = percentile_lower[:-1]
    
    # If mean is True, use mean forecast values instead of percentile intervals
    if mean:
        mean_forecast = np.mean(samples, axis=0)
        shifted_percentile_higher = mean_forecast[:-1]
        shifted_percentile_lower = mean_forecast[:-1]
        
    # Calculate log returns as percentages
    log_returns = np.log(shifted_actual_values / actual_test_values[:-1])
    log_returns = np.nan_to_num(log_returns, nan=0.0)

    # Determine conditions for trading decisions based on custom percentiles
    higher_correct_condition_custom = (shifted_percentile_lower > actual_test_values[:-1]) & (shifted_actual_values > actual_test_values[:-1])
    lower_correct_condition_custom = (shifted_percentile_higher < actual_test_values[:-1]) & (shifted_actual_values < actual_test_values[:-1])
    overall_correct_condition_custom = higher_correct_condition_custom | lower_correct_condition_custom

    # Generate positions based on custom strategy (1 for long, -1 for short, 0 otherwise)
    positions_custom = np.where(higher_correct_condition_custom, 1, np.where(lower_correct_condition_custom, -1, 0))

    # Calculate total profit using log returns
    profit_custom = np.sum(positions_custom * log_returns)

    # Calculate Sharpe ratio (assuming risk-free rate is 0)
    sharpe_custom = np.mean(positions_custom * log_returns) / np.std(log_returns) if np.std(log_returns) != 0 else 0

    # Return results as a dictionary
    return {
        "custom": {
            "overall_correct_percentage": 100 * np.sum(overall_correct_condition_custom) / len(shifted_actual_values),
            "profit": profit_custom,
            "sharpe_ratio": sharpe_custom,
        }
    }

def evaluate_trading_framework(samples, y_test):
    """
    Evaluates a trading framework across a range of percentiles and compares performance.
    
    Args:
        samples (numpy.ndarray): The predicted samples for each time step.
        y_test (torch.Tensor or numpy.ndarray): The actual test values.
    
    Returns:
        tuple: Dictionaries containing overall correct percentages and Sharpe ratios for each percentile.
    """
    # Define the range of lower percentiles to evaluate
    lower_percentiles = [0.1, 1, 5] + list(range(10, 100, 5)) + [99, 99.9]
    overall_correct_percentages_custom = {}
    sharpe_ratios_custom = {}
    
    # Evaluate for each lower percentile and store the results
    for perc in lower_percentiles:
        results = evaluate_trading_framework_component(samples, y_test, 100 - perc, perc)

        # Store the metrics for the custom strategy
        overall_correct_percentages_custom[perc] = results["custom"]["overall_correct_percentage"]
        sharpe_ratios_custom[perc] = results["custom"]["sharpe_ratio"]
    
    # Evaluate using the mean strategy for reference
    results = evaluate_trading_framework_component(samples, y_test, 10, 90, mean=True)

    # Store the metrics for the mean strategy
    overall_correct_percentages_custom["mean"] = results["custom"]["overall_correct_percentage"]
    sharpe_ratios_custom["mean"] = results["custom"]["sharpe_ratio"]

    return overall_correct_percentages_custom, sharpe_ratios_custom