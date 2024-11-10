import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

import pickle
import os

# allow duplication of parallel libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

##############################################################################
# Combine outputs

# List of filenames to combine
filenames = [
    'output/output_0_17.pkl', 
    'output/output_17_34.pkl', 
    'output/output_34_51.pkl', 
    'output/output_51_68.pkl', 
    'output/output_68_84.pkl', 
    'output/output_84_100.pkl'
]

# Initialize combined dictionaries
combined_data = {
    "vae_train_mse_runs": {},
    "vae_test_mse_runs": {},
    "cnn_lstm_train_mse_runs": {},
    "cnn_lstm_test_mse_runs": {},
    "vae_train_mse_with_original": {},
    "vae_test_mse_with_original": {},
    "cnn_lstm_train_mse_with_original": {},
    "cnn_lstm_test_mse_with_original": {},
    "latent_stats": {},
    "vae_trade_stat": {}
}

# Function to merge dictionaries
def merge_dicts(main_dict, new_dict):
    for key, value in new_dict.items():
        if key in main_dict:
            # If it's a dictionary, we need to merge recursively
            if isinstance(value, dict):
                merge_dicts(main_dict[key], value)
            # Check if both are lists before extending
            elif isinstance(main_dict[key], list) and isinstance(value, list):
                main_dict[key].extend(value)  # Combine lists
            else:
                # For non-list values, store the latest value
                main_dict[key] = value
        else:
            main_dict[key] = value

# Loop over each file and merge the data
for filename in filenames:
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        for key in combined_data.keys():
            merge_dicts(combined_data[key], data[key])

# Save the combined dictionary
with open('output/combined_output.pkl', 'wb') as file:
    pickle.dump(combined_data, file)

print("All files combined successfully into 'combined_output.pkl'.")
##############################################################################

N = 100
num_epochs = 10000
window_size = 50
noise_ratios = [0.1, 0.5, 1, 2, 4, 8, 16, 32]
report_epochs = np.array(np.linspace(100,num_epochs,int(num_epochs/100)), dtype=int)
output_filename = 'output/combined_output.pkl'

################################################################################
# Function to load the data from the pickle file and unpack into variables
def load_and_unpack(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    # Unpack the dictionary items into variables with the same names
    for key, value in data.items():
        globals()[key] = value

    print(f"Data has been unpacked from {filename}.")

load_and_unpack(output_filename)

###############################################################################

################################################################################
"""Overview of MSE (Std. Err.) for TempVAE and CNN-LSTM with respect to the 
target variable across different noise ratios."""
################################################################################

epoch_ = 5000

# Function to calculate mean and standard error for a given epoch
def calculate_mean_and_std_error(mse_runs, epoch):
    means = {nr: np.nanmean(mse_runs[nr][epoch]) for nr in noise_ratios}
    std_errors = {nr: np.nanstd(mse_runs[nr][epoch]) / np.sqrt(N) for nr in noise_ratios}
    return means, std_errors

# Calculate statistics for epoch 200
vae_train_means, vae_train_std_errors = calculate_mean_and_std_error(vae_train_mse_runs, epoch_)
vae_test_means, vae_test_std_errors = calculate_mean_and_std_error(vae_test_mse_runs, epoch_)
cnn_lstm_train_means, cnn_lstm_train_std_errors = calculate_mean_and_std_error(cnn_lstm_train_mse_runs, epoch_)
cnn_lstm_test_means, cnn_lstm_test_std_errors = calculate_mean_and_std_error(cnn_lstm_test_mse_runs, epoch_)

# Function to format numbers according to the specified rules
def format_number(num, is_std_error=False):
    if num <= 0:
        formatted_num = '0.00'
    elif num < 1e-2:
        exponent = int(np.floor(np.log10(num)))
        if is_std_error:
            formatted_num = f'$({{<10^{{{exponent}}}}})$'  # Parentheses inside math mode
        else:
            formatted_num = f'$<10^{{{exponent}}}$'
    else:
        if is_std_error:
            formatted_num = f'({num:.2f})'
        else:
            formatted_num = f'{num:.2f}'
    return formatted_num

# Function to prepare values for LaTeX and apply bolding
def prepare_values(model_means, other_model_means, is_std_error=False):
    values = []
    for nr in noise_ratios:
        val = format_number(model_means[nr], is_std_error)
        if not is_std_error:
            # Compare with the other model's mean MSE
            other_val = other_model_means[nr]
            # Apply bold formatting if this model has lower MSE
            if model_means[nr] < other_val:
                val = f'\\textbf{{{val}}}'
        values.append(val)
    return values

#######################
# FILL IN TABLE 1
#######################

# Code to populate table1.txt
with open('output/table1.txt', 'w') as f:
    # Write the LaTeX table header
    f.write('\\begin{longtable}{c c c c c c c c c c}\n')
    f.write('\\caption{\\footnotesize{Overview of $MSE$ with standard errors for TempVAE and CNN-LSTM with respect to the target variable ($\\tilde{y}_t$) across different noise ratios ($\\alpha$).}} \\label{mse_table_combined} \\\\\n')
    f.write('\\toprule\n')
    f.write('\\multirow{2}{*}{\\textbf{Data}} & \\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{8}{c}{\\textbf{Noise Ratio} ($\\alpha$)} \\\\\n')
    f.write('\\cmidrule(lr){3-10}\n')
    f.write(' & & ' + ' & '.join('{:.2f}'.format(nr) for nr in noise_ratios) + ' \\\\\n')
    f.write('\\midrule\n')
    f.write('\\endfirsthead\n\n')
    f.write('\\caption[]{(continued)} \\\\\n')
    f.write('\\toprule\n')
    f.write('\\multirow{2}{*}{\\textbf{Data}} & \\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{8}{c}{\\textbf{Noise Ratio} ($\\alpha$)} \\\\\n')
    f.write('\\cmidrule(lr){3-10}\n')
    f.write(' & & ' + ' & '.join('{:.2f}'.format(nr) for nr in noise_ratios) + ' \\\\\n')
    f.write('\\midrule\n')
    f.write('\\endhead\n\n')
    f.write('\\midrule \\multicolumn{10}{r}{{Continued on next page}} \\\\\n')
    f.write('\\endfoot\n\n')
    f.write('\\bottomrule\n')
    f.write('\\endlastfoot\n\n')

    # Write Train data for CNN-LSTM
    f.write('\\multirow{4}{*}{Train} & \\multirow{2}{*}{CNN-LSTM} & ')
    # Prepare values with bolding where applicable
    cnn_lstm_train_means_values = prepare_values(cnn_lstm_train_means, vae_train_means)
    f.write(' & '.join(cnn_lstm_train_means_values) + ' \\\\\n')
    f.write(' & & ')
    cnn_lstm_train_std_errors_values = prepare_values(cnn_lstm_train_std_errors, vae_train_std_errors, is_std_error=True)
    f.write(' & '.join(cnn_lstm_train_std_errors_values) + ' \\\\\n')
    f.write('\\cmidrule(lr){2-10}\n')

    # Write Train data for TempVAE
    f.write(' & \\multirow{2}{*}{TempVAE} & ')
    # Prepare values with bolding where applicable
    vae_train_means_values = prepare_values(vae_train_means, cnn_lstm_train_means)
    f.write(' & '.join(vae_train_means_values) + ' \\\\\n')
    f.write(' & & ')
    vae_train_std_errors_values = prepare_values(vae_train_std_errors, cnn_lstm_train_std_errors, is_std_error=True)
    f.write(' & '.join(vae_train_std_errors_values) + ' \\\\\n')
    f.write('\\midrule\n')

    # Write Test data for CNN-LSTM
    f.write('\\multirow{4}{*}{Test} & \\multirow{2}{*}{CNN-LSTM} & ')
    # Prepare values with bolding where applicable
    cnn_lstm_test_means_values = prepare_values(cnn_lstm_test_means, vae_test_means)
    f.write(' & '.join(cnn_lstm_test_means_values) + ' \\\\\n')
    f.write(' & & ')
    cnn_lstm_test_std_errors_values = prepare_values(cnn_lstm_test_std_errors, vae_test_std_errors, is_std_error=True)
    f.write(' & '.join(cnn_lstm_test_std_errors_values) + ' \\\\\n')
    f.write('\\cmidrule(lr){2-10}\n')

    # Write Test data for TempVAE
    f.write(' & \\multirow{2}{*}{TempVAE} & ')
    # Prepare values with bolding where applicable
    vae_test_means_values = prepare_values(vae_test_means, cnn_lstm_test_means)
    f.write(' & '.join(vae_test_means_values) + ' \\\\\n')
    f.write(' & & ')
    vae_test_std_errors_values = prepare_values(vae_test_std_errors, cnn_lstm_test_std_errors, is_std_error=True)
    f.write(' & '.join(vae_test_std_errors_values) + ' \\\\\n')
    f.write('\\end{longtable}\n')


################################################################################
# FILL IN Figure 1
################################################################################

# Set the Seaborn style for academic plots
sns.set_context("notebook", font_scale=1.5)
sns.set_style("whitegrid")

# Function to plot Train and Test MSE with standard error for all epochs
def plot_mse_with_std_error(noise_ratio, train_mse_runs, test_mse_runs, label_prefix):
    # Get all epochs up to 5000 and sort them
    epochs = sorted(epoch for epoch in train_mse_runs[noise_ratio].keys() if epoch <= 5000)
    
    # Calculate means and standard errors for Train and Test MSE
    train_means = [np.nanmean(train_mse_runs[noise_ratio][epoch]) for epoch in epochs]
    train_std_errors = [np.nanstd(train_mse_runs[noise_ratio][epoch]) / np.sqrt(N) for epoch in epochs]
    test_means = [np.nanmean(test_mse_runs[noise_ratio][epoch]) for epoch in epochs]
    test_std_errors = [np.nanstd(test_mse_runs[noise_ratio][epoch]) / np.sqrt(N) for epoch in epochs]
    
    # Plot Train MSE with standard error
    plt.plot(epochs, train_means, label=f'{label_prefix} Train', linestyle='-', marker='o')
    plt.fill_between(epochs, 
                     np.array(train_means) - np.array(train_std_errors), 
                     np.array(train_means) + np.array(train_std_errors), 
                     alpha=0.2)

    # Plot Test MSE with standard error
    plt.plot(epochs, test_means, label=f'{label_prefix} Test', linestyle='--', marker='x')
    plt.fill_between(epochs, 
                     np.array(test_means) - np.array(test_std_errors), 
                     np.array(test_means) + np.array(test_std_errors), 
                     alpha=0.2)

    # Enhance plot aesthetics
    plt.title(f'MSE vs Epoch for $\\alpha$ (noise ratio) = {noise_ratio}', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('$MSE$')
    plt.legend()
    plt.tight_layout()

# Plot graphs for each noise factor
for x, noise_ratio in enumerate([0.1, 1, 4, 32]):
    plt.figure(figsize=(12, 6))
    plt.title(f'MSE vs Epoch for Noise Ratio {noise_ratio}', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    
    # Get all epochs up to 5000
    epochs = [epoch for epoch in report_epochs if epoch <= 5000]
    
    # Calculate MSE values for the current noise ratio
    vae_train_mse = [np.nanmean(vae_train_mse_runs[noise_ratio][epoch]) for epoch in epochs]
    vae_test_mse = [np.nanmean(vae_test_mse_runs[noise_ratio][epoch]) for epoch in epochs]
    cnn_lstm_train_mse = [np.nanmean(cnn_lstm_train_mse_runs[noise_ratio][epoch]) for epoch in epochs]
    cnn_lstm_test_mse = [np.nanmean(cnn_lstm_test_mse_runs[noise_ratio][epoch]) for epoch in epochs]

    # Combine all MSE values to find the 90th percentile
    all_mse_values = vae_train_mse + vae_test_mse + cnn_lstm_train_mse + cnn_lstm_test_mse
    upper_y_lim = np.quantile(all_mse_values, 0.85 + (x + 1) / 90)
    
    # Set y-axis limit from 0 to the xth percentile and x-axis limit up to 5000 epochs
    plt.ylim([0, upper_y_lim])
    plt.xlim([0, 5000])
    plt.xticks(ticks=np.arange(0, 5001, 500))
    
    # Plot for TempVAE
    plot_mse_with_std_error(noise_ratio, vae_train_mse_runs, vae_test_mse_runs, 'TempVAE')

    # Plot for CNN-LSTM
    plot_mse_with_std_error(noise_ratio, cnn_lstm_train_mse_runs, cnn_lstm_test_mse_runs, 'CNN-LSTM')

    # Add a horizontal line for the True MSE (equal to noise_ratio * 100)
    true_mse = noise_ratio * 100
    plt.axhline(y=true_mse, color='black', linestyle='-', linewidth=1.5, label='True MSE')

    # Add legend and grid
    plt.legend(loc='upper right')
    plt.grid(visible=True, linestyle='--', linewidth=0.7)
    plt.savefig(f'output/mse_vs_epoch_{int(noise_ratio*100)}.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

################################################################################
# FILL IN TABLE 2
################################################################################

# Define the noise ratios to include in Table 2 (all)
table2_noise_ratios = [nr for nr in noise_ratios if nr >= 0]

# Function to calculate mean and standard error for a given epoch
def calculate_mean_and_std_error(mse_runs, epoch, noise_ratios):
    means = {nr: np.nanmean(mse_runs[nr][epoch]) for nr in noise_ratios}
    std_errors = {nr: np.nanstd(mse_runs[nr][epoch]) / np.sqrt(N) for nr in noise_ratios}
    return means, std_errors

# Calculate statistics for epoch 200
vae_train_means, vae_train_std_errors = calculate_mean_and_std_error(vae_train_mse_with_original, epoch_, table2_noise_ratios)
vae_test_means, vae_test_std_errors = calculate_mean_and_std_error(vae_test_mse_with_original, epoch_, table2_noise_ratios)
cnn_lstm_train_means, cnn_lstm_train_std_errors = calculate_mean_and_std_error(cnn_lstm_train_mse_with_original, epoch_, table2_noise_ratios)
cnn_lstm_test_means, cnn_lstm_test_std_errors = calculate_mean_and_std_error(cnn_lstm_test_mse_with_original, epoch_, table2_noise_ratios)

# Function to prepare values for LaTeX and apply bolding
def prepare_values(model_means, other_model_means, is_std_error=False):
    values = []
    for nr in table2_noise_ratios:
        val = format_number(model_means[nr], is_std_error)
        if not is_std_error:
            # Compare with the other model's mean MSE
            other_val = other_model_means[nr]
            # Apply bold formatting if this model has lower MSE
            if model_means[nr] < other_val:
                val = f'\\textbf{{{val}}}'
        values.append(val)
    return values

# Code to populate table2.txt
with open('output/table2.txt', 'w') as f:
    # Write the LaTeX table header
    f.write('\\begin{longtable}{c c c c c c c c c c c c}\n')
    f.write('\\caption{\\footnotesize{Overview of Train and Test $MSE$ versus \\textit{true} signal ($y_t$) with standard errors for TempVAE and CNN-LSTM across different noise ratios ($\\alpha$).}} \\label{mse_table_true} \\\\\n')
    f.write('\\toprule\n')
    f.write('\\multirow{2}{*}{\\textbf{Data}} & \\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{8}{c}{\\textbf{Noise Ratio} ($\\alpha$)} \\\\\n')
    f.write('\\cmidrule(lr){3-10}\n')
    f.write(' & & ' + ' & '.join('{:.2f}'.format(nr) for nr in table2_noise_ratios) + ' \\\\\n')
    f.write('\\midrule\n')
    f.write('\\endfirsthead\n\n')
    f.write('\\caption[]{(continued)} \\\\\n')
    f.write('\\toprule\n')
    f.write('\\multirow{2}{*}{\\textbf{Data}} & \\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{8}{c}{\\textbf{Noise Ratio} ($\\alpha$)} \\\\\n')
    f.write('\\cmidrule(lr){3-10}\n')
    f.write(' & & ' + ' & '.join('{:.2f}'.format(nr) for nr in table2_noise_ratios) + ' \\\\\n')
    f.write('\\midrule\n')
    f.write('\\endhead\n\n')
    f.write('\\midrule \\multicolumn{10}{r}{{Continued on next page}} \\\\\n')
    f.write('\\endfoot\n\n')
    f.write('\\bottomrule\n')
    f.write('\\endlastfoot\n\n')

    # Write Train data for CNN-LSTM
    f.write('\\multirow{4}{*}{Train} & \\multirow{2}{*}{CNN-LSTM} & ')
    # Prepare values with bolding where applicable
    cnn_lstm_train_means_values = prepare_values(cnn_lstm_train_means, vae_train_means)
    f.write(' & '.join(cnn_lstm_train_means_values) + ' \\\\\n')
    f.write(' & & ')
    cnn_lstm_train_std_errors_values = prepare_values(cnn_lstm_train_std_errors, vae_train_std_errors, is_std_error=True)
    f.write(' & '.join(cnn_lstm_train_std_errors_values) + ' \\\\\n')
    f.write('\\cmidrule(lr){2-10}\n')

    # Write Train data for TempVAE
    f.write(' & \\multirow{2}{*}{TempVAE} & ')
    # Prepare values with bolding where applicable
    vae_train_means_values = prepare_values(vae_train_means, cnn_lstm_train_means)
    f.write(' & '.join(vae_train_means_values) + ' \\\\\n')
    f.write(' & & ')
    vae_train_std_errors_values = prepare_values(vae_train_std_errors, cnn_lstm_train_std_errors, is_std_error=True)
    f.write(' & '.join(vae_train_std_errors_values) + ' \\\\\n')
    f.write('\\midrule\n')

    # Write Test data for CNN-LSTM
    f.write('\\multirow{4}{*}{Test} & \\multirow{2}{*}{CNN-LSTM} & ')
    # Prepare values with bolding where applicable
    cnn_lstm_test_means_values = prepare_values(cnn_lstm_test_means, vae_test_means)
    f.write(' & '.join(cnn_lstm_test_means_values) + ' \\\\\n')
    f.write(' & & ')
    cnn_lstm_test_std_errors_values = prepare_values(cnn_lstm_test_std_errors, vae_test_std_errors, is_std_error=True)
    f.write(' & '.join(cnn_lstm_test_std_errors_values) + ' \\\\\n')
    f.write('\\cmidrule(lr){2-10}\n')

    # Write Test data for TempVAE
    f.write(' & \\multirow{2}{*}{TempVAE} & ')
    # Prepare values with bolding where applicable
    vae_test_means_values = prepare_values(vae_test_means, cnn_lstm_test_means)
    f.write(' & '.join(vae_test_means_values) + ' \\\\\n')
    f.write(' & & ')
    vae_test_std_errors_values = prepare_values(vae_test_std_errors, cnn_lstm_test_std_errors, is_std_error=True)
    f.write(' & '.join(vae_test_std_errors_values) + ' \\\\\n')
    f.write('\\end{longtable}\n')


################################################################################
# FILL IN Figure 1 of Results
################################################################################

# Set the Seaborn style for academic plots
sns.set_context("notebook", font_scale=1.5)
sns.set_style("whitegrid")

# Define the noise ratios and percentiles for plotting
noise_ratios = [0.1, 1, 4, 32]
lower_percentiles = [0.1, 1, 5] + list(range(10, 100, 5)) + [99, 99.9]

# Initialize dictionaries to store the aggregated results for each noise ratio
overall_correct_percentages = {nr: [] for nr in noise_ratios}
sharpe_ratios = {nr: [] for nr in noise_ratios}
overall_correct_std_error = {nr: [] for nr in noise_ratios}
sharpe_std_error = {nr: [] for nr in noise_ratios}

# Aggregate the data across all runs
for noise_ratio in noise_ratios:
    temp_correct_percentages = []
    temp_sharpe_ratios = []
    for run in range(N):
        # Check if the current run and noise ratio exist and contain data
        if not vae_trade_stat.get(run, {}).get(noise_ratio):
            continue
        
        # Extract values for each lower percentile if they exist
        correct_percentages = [
            vae_trade_stat[run][noise_ratio].get('Perc_profitable', {}).get(lower_percentile, np.nan)
            for lower_percentile in lower_percentiles
        ]
        sharpe_values = [
            vae_trade_stat[run][noise_ratio].get('SR', {}).get(lower_percentile, np.nan)
            for lower_percentile in lower_percentiles
        ]

        # Append only if no missing data
        if not any(np.isnan(correct_percentages)) and not any(np.isnan(sharpe_values)):
            temp_correct_percentages.append(correct_percentages)
            temp_sharpe_ratios.append(sharpe_values)

    # Calculate the average and standard error over all runs for each noise ratio
    if temp_correct_percentages:  # Ensure there's data to calculate
        overall_correct_percentages[noise_ratio] = np.nanmean(temp_correct_percentages, axis=0)
        overall_correct_std_error[noise_ratio] = np.nanstd(temp_correct_percentages, axis=0) / np.sqrt(len(temp_correct_percentages))
    
    if temp_sharpe_ratios:  # Ensure there's data to calculate
        sharpe_ratios[noise_ratio] = np.nanmean(temp_sharpe_ratios, axis=0)
        sharpe_std_error[noise_ratio] = np.nanstd(temp_sharpe_ratios, axis=0) / np.sqrt(len(temp_sharpe_ratios))

# Plot the Overall Percentage Correct vs Lower Percentile for each noise ratio
plt.figure(figsize=(12, 6))
for noise_ratio in noise_ratios:
    sns.lineplot(x=lower_percentiles, y=overall_correct_percentages[noise_ratio], marker='o', label=f'{noise_ratio}')
    plt.fill_between(
        lower_percentiles,
        overall_correct_percentages[noise_ratio] - overall_correct_std_error[noise_ratio],
        overall_correct_percentages[noise_ratio] + overall_correct_std_error[noise_ratio],
        alpha=0.2
    )
plt.title('Overall Percentage Correct vs Lower Percentile ($x$) of TempVAE', fontweight='bold')
plt.xlabel('Lower Percentile ($x$)')
plt.ylabel('Overall Percentage Correct (%)')
plt.xticks(ticks=np.arange(0, 101, 10))
plt.xlim([0, 100])
plt.legend(title='$\\alpha$', loc='upper left')
plt.grid(visible=True, linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig('output/overall_corr_percentage.png', format='png', dpi=300, bbox_inches='tight')
plt.show()


# Plot the Sharpe Ratio vs Lower Percentile for each noise ratio
plt.figure(figsize=(12, 6))
for noise_ratio in noise_ratios:
    sns.lineplot(x=lower_percentiles, y=sharpe_ratios[noise_ratio], marker='o', label=f'{noise_ratio}')
    plt.fill_between(
        lower_percentiles,
        sharpe_ratios[noise_ratio] - sharpe_std_error[noise_ratio],
        sharpe_ratios[noise_ratio] + sharpe_std_error[noise_ratio],
        alpha=0.2
    )
    
plt.title('Sharpe Ratio vs Lower Percentile ($x$) of TempVAE', fontweight='bold')
plt.xlabel('Lower Percentile ($x$)')
plt.ylabel('Average Sharpe Ratio')
plt.xticks(ticks=np.arange(0, 101, 10))
plt.xlim([0, 100])
plt.legend(title='$\\alpha$', loc='upper left')
plt.grid(visible=True, linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig('output/sharpe.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

################################################################################
# Latent Variable Analysis
################################################################################

#######################
# PLOT 1 - ACF plots
#######################

# Set seaborn style
sns.set(style='whitegrid')

# Define the noise ratios as a variable
noise_ratios = [0.1, 0.5, 1, 4, 32]

# Initialize dictionaries to store ACF values for each noise ratio
train_acf_all_runs = {nr: [] for nr in noise_ratios}
test_acf_all_runs = {nr: [] for nr in noise_ratios}

# Calculate ACF for each run and noise ratio
for run in range(N):
    for noise_ratio in noise_ratios:
        # Check if the current run and noise ratio exist and contain data
        if not latent_stats.get(run, {}).get(noise_ratio):
            continue
        
        # Grab train and test latent variables if available
        train_data = latent_stats[run][noise_ratio].get('train')
        test_data = latent_stats[run][noise_ratio].get('test')
        
        if train_data is None or test_data is None:
            continue  # Skip if either train or test data is missing
        
        # Extract the latent variables and calculate the mean
        train_z = train_data[0]
        avg_train_z = np.mean(train_z, axis=1)

        test_z = test_data[0]
        avg_test_z = np.mean(test_z, axis=1)

        # Calculate the ACF values for the train set
        acf_values_train = acf(avg_train_z, nlags=10, alpha=0.05)
        train_acf_vals, _ = acf_values_train
        train_acf_all_runs[noise_ratio].append(train_acf_vals)

        # Calculate the ACF values for the test set
        acf_values_test = acf(avg_test_z, nlags=10, alpha=0.05)
        test_acf_vals, _ = acf_values_test
        test_acf_all_runs[noise_ratio].append(test_acf_vals)

# Function to plot ACF with standard error
def plot_acf_with_error(acf_data, title, name):
    lags = np.arange(0, 11)  # Lags from 1 to 11

    plt.figure(figsize=(12, 6))

    # Plot ACF for each noise ratio
    for noise_ratio in noise_ratios:
        # Convert lists to numpy arrays for easier computation
        acf_array = np.array(acf_data[noise_ratio])
        
        # Calculate mean and standard error
        mean_acf = np.mean(acf_array, axis=0)
        std_error_acf = np.std(acf_array, axis=0) / np.sqrt(acf_array.shape[0])
        
        # Plot the ACF values with standard error
        plt.plot(lags, mean_acf, label=f'{noise_ratio}', marker='x')
        plt.fill_between(lags, mean_acf - std_error_acf, mean_acf + std_error_acf, alpha=0.2)

    # Configure plot settings
    plt.title(title, fontweight='bold')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation Coefficient')
    plt.ylim(0, 1)
    plt.xlim(1, 10)
    plt.xticks(ticks=np.arange(0, 10), labels=np.arange(0, 10))
    plt.yticks(ticks=np.arange(1, 11)/10, labels=np.arange(1, 11)/10)
    plt.legend(title='$\\alpha$')
    plt.grid(True)
    plt.savefig(f'output/acf_{name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot for Train ACF
plot_acf_with_error(train_acf_all_runs, 'Train ACF for Different Noise Ratios ($\\alpha$)', 'train')

# Plot for Test ACF
plot_acf_with_error(test_acf_all_runs, 'Test ACF for Different Noise Ratios ($\\alpha$)', 'test')