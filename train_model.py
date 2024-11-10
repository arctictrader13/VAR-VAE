import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import argparse

import pickle
import os
import time

# import the models
from TempVAE import TempVAE
from CNN_LSTM import CNN_LSTM

# import evaluation of trading
from trading_eval import evaluate_trading_framework

# allow duplication of parallel libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set up argument parser
parser = argparse.ArgumentParser(description="Train models with specified run range.")
parser.add_argument('--run', nargs=2, type=int, help="Specify the range of runs (inclusive). Example: --run 0 5")
args = parser.parse_args()
start_run, end_run = args.run

###############################################################################
# Generate data
###############################################################################
def gen_data(noise_ratio):
    short_term = 0.5
    long_term = 1 - short_term

    # Time vector
    t = np.linspace(0, 100, 2000)  # 0 to 10 seconds with 1000 samples

    # Short-term signal: high-frequency sine wave
    short_term_signal = 0.5 * np.sin(10 * np.pi * t)  # Frequency is 5 Hz

    # Long-term signal: low-frequency sine wave
    long_term_signal = 1.0 * np.sin(np.pi * t)  # Frequency is 0.5 Hz, amplitude is larger

    # Linear combination of short-term and long-term signals
    sin_wave = short_term * short_term_signal + long_term * long_term_signal
    
    # scale by constant to make sure absolute median signal is close to 1
    sin_wave = 100 * (sin_wave + 1)
    
    # Calculate the median of the sinusoidal wave
    sin_mean = np.median(np.abs(sin_wave))

    # Noise variance is median(sin_wave) * noise_ratio
    noise_variance = sin_mean * noise_ratio
    noise_std_dev = np.sqrt(noise_variance)  # Standard deviation for normal distribution

    # Generate noise with the calculated variance
    noise = np.random.normal(0, noise_std_dev, len(sin_wave))

    # Add noise to the sinusoidal wave
    noisy_signal = sin_wave + noise

    return t, sin_wave, noisy_signal

###############################################################################
# Prepare data using a sliding window approach
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

###############################################################################

# Number of independent runs
N = 100

# Initialize training parameters
window_size = 50
num_epochs = 10000
noise_ratios = [0.1, 0.5, 1, 2, 4, 8, 16, 32]
report_epochs = np.array(np.linspace(100,num_epochs,int(num_epochs/100)), dtype=int)
num_samples_trading_eval = 100
output_filename = 'output.pkl'

# Initialize storage for Table 1 statistics across N runs
vae_train_mse_runs = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}
vae_test_mse_runs = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}
cnn_lstm_train_mse_runs = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}
cnn_lstm_test_mse_runs = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}


# Initialize storage for Table 2 statistics across N runs
vae_train_mse_with_original = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}
vae_test_mse_with_original = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}
cnn_lstm_train_mse_with_original = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}
cnn_lstm_test_mse_with_original = {nr: {epoch: [] for epoch in report_epochs} for nr in noise_ratios}

# Initialize storage for latent variables
latent_stats = {run: {nr: {} for nr in noise_ratios} for run in range(N)}


# Initialize storage for Trading Evaluation across N runs
vae_trade_stat = {run: {nr: {} for nr in noise_ratios} for run in range(N)}


# Function to save the current state of all data structures to a pickle file
def save_data_to_pickle(filename, vae_train_mse_runs, vae_test_mse_runs, cnn_lstm_train_mse_runs, cnn_lstm_test_mse_runs,
                        vae_train_mse_with_original, vae_test_mse_with_original, cnn_lstm_train_mse_with_original, 
                        cnn_lstm_test_mse_with_original, latent_stats, vae_trade_stat):
    data = {
        "vae_train_mse_runs": vae_train_mse_runs,
        "vae_test_mse_runs": vae_test_mse_runs,
        "cnn_lstm_train_mse_runs": cnn_lstm_train_mse_runs,
        "cnn_lstm_test_mse_runs": cnn_lstm_test_mse_runs,
        "vae_train_mse_with_original": vae_train_mse_with_original,
        "vae_test_mse_with_original": vae_test_mse_with_original,
        "cnn_lstm_train_mse_with_original": cnn_lstm_train_mse_with_original,
        "cnn_lstm_test_mse_with_original": cnn_lstm_test_mse_with_original,
        "latent_stats": latent_stats,
        "vae_trade_stat": vae_trade_stat
    }
    
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    
    
# Run N independent training processes
for run in range(N):
    # Skip runs that are not in the specified range
    if run < start_run or run > end_run:
        continue
        
    print(f"Starting run {run + 1}/{N}")
    
    start_time = time.time()  # Record the start time
    for noise_ratio in noise_ratios:

        # Generate data for the given noise ratio
        t, sin_wave, noisy_sin_wave = gen_data(noise_ratio)
        
        # Create dataset with sliding window
        X, y = create_dataset(noisy_sin_wave, window_size)
        
        # Split into training (70%) and test (30%) sets
        split_index = int(len(X) * 0.7)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Convert to torch tensors
        X_train = torch.tensor(X_train.reshape(-1, 1, window_size), dtype=torch.float32)
        y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).view(-1)
        X_test = torch.tensor(X_test.reshape(-1, 1, window_size), dtype=torch.float32)
        y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).view(-1)
        
        # Initialize models and optimizer
        tempvae_model = TempVAE(num_features=1, encoding_dim=32, window_size=window_size)
        cnn_lstm_model = CNN_LSTM(num_features=1, encoding_dim=32, window_size=window_size)
        
        criterion = nn.MSELoss()
        
        optimizer_vae = optim.Adam(tempvae_model.parameters(), lr=0.01)
        optimizer_cnn_lstm = optim.Adam(cnn_lstm_model.parameters(), lr=0.001)
        
        ######################################################################
        # Train TempVAE model
        ######################################################################
        
        for epoch in range(1, num_epochs + 1):
            tempvae_model.train()
            optimizer_vae.zero_grad()
            output_vae, std_dev, mean, z, z_hat = tempvae_model(X_train)
            
            # Compute loss
            reconstruction_loss_vae = criterion(output_vae.squeeze(), y_train)
            ###################################################################
            A1 = tempvae_model.fc2.bias  # Shape: [32]
            A2 = tempvae_model.fc2.weight  # Shape: [32, 32]
            
            # mean
            mean_reshaped = mean.view(1365, 32, 1)
            weighted_mean = torch.matmul(A2, mean_reshaped)
            bias_reshaped = A1.view(1, 32, 1)
            biased_weighted_mean = weighted_mean + bias_reshaped
            mean_new = biased_weighted_mean.view(1365, 1, 32)
            
            # variance
            var = std_dev ** 2
            
            # Reshape var to [1365, 32], removing the singleton dimension
            var_reshaped = var.view(1365, 32)
            var_diag = torch.diag_embed(var_reshaped)
            A2_expanded = A2.unsqueeze(0)  # Shape becomes [1, 32, 32]
            variance_weighted = torch.matmul(A2_expanded, torch.matmul(var_diag, A2_expanded.transpose(-1, -2)))
            variance_weighted_diag = torch.diagonal(variance_weighted, dim1=-2, dim2=-1)
            
            # Reshape back to [1365, 1, 32] like the original var
            variance_weighted_final = variance_weighted_diag.view(1365, 1, 32)                        
            ###################################################################
            # model outputs standard deviation not log variance
            log_var = (std_dev**2).log()
            
            kl_loss_vae = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            var_loss = criterion(z_hat[:-1, :, :], z[1:, :, :])
            
            loss_vae = reconstruction_loss_vae +  kl_loss_vae + var_loss
            
            # Backward and optimize
            loss_vae.backward()
            optimizer_vae.step()

            # Print train and test losses every 100 epochs
            if epoch % 100 == 0:
                tempvae_model.eval()
                with torch.no_grad():
                    # Calculate Train Loss
                    forecasted_train_vae, _, _, _, _ = tempvae_model(X_train)
                    train_loss = criterion(forecasted_train_vae.squeeze(), y_train).item()
                    
                    # Calculate Test Loss
                    forecasted_test_vae, _, _, _, _ = tempvae_model(X_test)
                    test_loss = criterion(forecasted_test_vae.squeeze(), y_test).item()
                    
                    #print(f"Run {run + 1}, Noise {noise_ratio}, TempVAE | Epoch {epoch} | Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                    #print(torch.exp(log_var).mean())
                    
            # Evaluate and store errors at report epochs
            if epoch in report_epochs:
                tempvae_model.eval()
                with torch.no_grad():
                    # Calculate Train MSE
                    forecasted_train_vae, _, _, _, _ = tempvae_model(X_train)
                    train_mse = criterion(forecasted_train_vae.squeeze(), y_train).item()
                    vae_train_mse_runs[noise_ratio][epoch].append(train_mse)
                    
                    # Calculate Test MSE
                    forecasted_test_vae, _, _, _, _ = tempvae_model(X_test)
                    test_mse = criterion(forecasted_test_vae.squeeze(), y_test).item()
                    vae_test_mse_runs[noise_ratio][epoch].append(test_mse)
                    
                    # Calculate MSE between original sin_wave and the forecast for TempVAE (Train)
                    vae_train_mse = criterion(forecasted_train_vae.squeeze(), torch.tensor(sin_wave[:len(y_train)], dtype=torch.float32)).item()
                    vae_train_mse_with_original[noise_ratio][epoch].append(vae_train_mse)
                    
                    # Calculate MSE between original sin_wave and the forecast for TempVAE (Test)
                    vae_test_mse = criterion(forecasted_test_vae.squeeze(), torch.tensor(sin_wave[len(sin_wave) - len(y_test):], dtype=torch.float32)).item()
                    vae_test_mse_with_original[noise_ratio][epoch].append(vae_test_mse)
        
        
        ######################################################################
        # Store latent variables to analyze later
        ######################################################################
        tempvae_model.eval()
        with torch.no_grad():
            # Calculate Train MSE
            _, _, _, z_train, z_hat_train = tempvae_model(X_train)
            
            # Calculate Test MSE
            _, _, _, z_test, z_hat_test = tempvae_model(X_test)
            
            latent_stats[run][noise_ratio] = {"train" : [z_train.detach().numpy()[:, 0, :], z_hat_train.detach().numpy()[:, 0, :]],
                                              "test" : [z_test.detach().numpy()[:, 0, :], z_hat_test.detach().numpy()[:, 0, :]]}
            
        ######################################################################
        # Check TempVAE model in trading
        ######################################################################
        
        # Generate multiple samples for each time step as a forecast
        samples = []
        tempvae_model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            for _ in range(num_samples_trading_eval):
                sampled_forecast, _, _, _, _ = tempvae_model(X_test)
                samples.append(sampled_forecast.squeeze().numpy())  # Collect sampled forecasts

        # Convert the list of samples to a numpy array (shape: [num_samples, len(X_test)])
        samples = np.array(samples)
            
        trade_stats = evaluate_trading_framework(samples, y_test) 
        vae_trade_stat[run][noise_ratio] = {"Perc_profitable" : trade_stats[0], "SR" : trade_stats[1]}
        
        ######################################################################
        # Train CNN-LSTM model
        ######################################################################
        for epoch in range(1, num_epochs + 1):
            cnn_lstm_model.train()
            optimizer_cnn_lstm.zero_grad()
            output_cnn_lstm = cnn_lstm_model(X_train)
            
            # Compute loss
            loss_cnn_lstm = criterion(output_cnn_lstm.squeeze(), y_train)
            
            # Backward and optimize
            loss_cnn_lstm.backward()
            optimizer_cnn_lstm.step()

            # Print train and test losses every 100 epochs
            if epoch % 100 == 0:
                cnn_lstm_model.eval()
                with torch.no_grad():
                    # Calculate Train Loss
                    forecasted_train_cnn_lstm = cnn_lstm_model(X_train)
                    train_loss = criterion(forecasted_train_cnn_lstm.squeeze(), y_train).item()
                    
                    # Calculate Test Loss
                    forecasted_test_cnn_lstm = cnn_lstm_model(X_test)
                    test_loss = criterion(forecasted_test_cnn_lstm.squeeze(), y_test).item()
                    
                    #print(f"Run {run + 1}, Noise {noise_ratio}, CNN-LSTM | Epoch {epoch} | Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Evaluate and store errors at report epochs
            if epoch in report_epochs:
                cnn_lstm_model.eval()
                with torch.no_grad():
                    # Calculate Train MSE
                    forecasted_train_cnn_lstm = cnn_lstm_model(X_train)
                    train_mse = criterion(forecasted_train_cnn_lstm.squeeze(), y_train).item()
                    cnn_lstm_train_mse_runs[noise_ratio][epoch].append(train_mse)
                    
                    # Calculate Test MSE
                    forecasted_test_cnn_lstm = cnn_lstm_model(X_test)
                    test_mse = criterion(forecasted_test_cnn_lstm.squeeze(), y_test).item()
                    cnn_lstm_test_mse_runs[noise_ratio][epoch].append(test_mse)
                    
                    # Calculate MSE between original sin_wave and the forecast for CNN-LSTM (Train)
                    cnn_lstm_train_mse = criterion(forecasted_train_cnn_lstm.squeeze(), torch.tensor(sin_wave[:len(y_train)], dtype=torch.float32)).item()
                    cnn_lstm_train_mse_with_original[noise_ratio][epoch].append(cnn_lstm_train_mse)
                    
                    # Calculate MSE between original sin_wave and the forecast for CNN-LSTM (Test)
                    cnn_lstm_test_mse = criterion(forecasted_test_cnn_lstm.squeeze(), torch.tensor(sin_wave[len(sin_wave) - len(y_test):], dtype=torch.float32)).item()
                    cnn_lstm_test_mse_with_original[noise_ratio][epoch].append(cnn_lstm_test_mse)
    
    #######################################################################
    # Store temporary results
    #######################################################################
    # Save the current state to a separate file for each run
    output_filename = f'output_{start_run}_{end_run}.pkl'
    
    save_data_to_pickle(output_filename, vae_train_mse_runs, vae_test_mse_runs, cnn_lstm_train_mse_runs, 
                        cnn_lstm_test_mse_runs, vae_train_mse_with_original, vae_test_mse_with_original, 
                        cnn_lstm_train_mse_with_original, cnn_lstm_test_mse_with_original, latent_stats, 
                        vae_trade_stat)
    
    elapsed_time = time.time() - start_time
    print(f"Run {run} took {elapsed_time:.2f} seconds")
