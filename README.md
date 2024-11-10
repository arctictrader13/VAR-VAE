# VAR-VAE: Temporal Variational Autoencoder for Time Series Forecasting

This repository contains the implementation of TempVAE, a generative model for time series forecasting that leverages Variational Autoencoders (VAEs) to transform complex time series data into a simpler latent structure based on a Vector Autoregression (VAR) process. This approach allows for effective forecasting, robust handling of noise, and the incorporation of uncertainty into predictions.

## Overview

### Files

- **`train_model.py`**: Script to train the TempVAE model. Runs are divided into chunks to allow incremental training.
- **`analysis.py`**: Script to analyze and combine results from different training runs. Generates various metrics, plots, and saves them to the output folder.
- **`CNN_LSTM.py`** and **`TempVAE.py`**: Implementations of the CNN-LSTM and TempVAE models, respectively.
- **Output**: Directory containing generated data, metrics, and plots for different noise ratios.

### Prerequisites

To run the code, install the required Python libraries:
```bash
pip install numpy torch matplotlib seaborn statsmodels
```

## Running the Code

### Training the Model

To train the model from scratch, execute the following commands in your terminal. Be aware that this process is computationally intensive and may take a significant amount of time (approximately 600 hours on a standard CPU).

```bash
python train_model.py --run 0 17
python train_model.py --run 17 34
python train_model.py --run 34 51
python train_model.py --run 51 68
python train_model.py --run 68 84
python train_model.py --run 84 100
```
Alternatively, you can skip training and run the analysis directly on the existing model outputs:
```bash
os.system("python analysis.py")
```
This will read from the previously saved training outputs and generate various evaluation metrics and plots in the **output** folder.

### Analysis Outputs
* **MSE Metrics:** Mean Squared Error (MSE) vs. epochs and noise ratios, comparing TempVAE to a baseline CNN-LSTM model.
* **Forecast Evaluation:** Sharpe ratio, percentage correct forecasts, and other metrics to assess the probabilistic forecasts of TempVAE.
* **Autocorrelation Analysis:** Autocorrelation function (ACF) plots to evaluate the structure of the latent space. 



