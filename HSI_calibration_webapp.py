# Import necessary libraries
# Streamlit for web app functionality, numpy and pandas for data manipulation,
# matplotlib and seaborn for static visualizations, plotly for interactive visuals,
# sklearn for machine learning and model evaluation, and scipy for signal processing.

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
import scipy.signal as signal
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as EVS
from sklearn.metrics import max_error as ME


# Give page description and short tutorial
# Provide a brief description and introduction to the web app.
st.markdown("# Hyperspectral calibration setup")
st.markdown("## Introduction")
st.markdown("This app is designed as a walkthrough to calibrate hyperspectral image data. Here is a general schema of the data processing steps:")
# st.image("https://i.imgur.com/1Z0ZQ2M.png", width = 500) - add image later
st.markdown("## 1. Load data")
st.markdown("Upload calibration data and sample data. "
            "For the previous version: Both should be a CSV-file with the first column containing the "
            "wavelengths and the remaining columns containing the samples (either calibration of unknown samples). "
            "It is important to remember that the CSV-file should have a header with the names of the samples. ")


# Load calibration data (df_calib) and sample data (df_sample)

# Prepare for data upload.
# The user can choose to upload a single file or multiple files.
input_format = st.radio("Select data format", ("Single file", "Multiple files"))

# Initialize an empty list to hold the names of the calibration files and an empty DataFrame to hold the calibration data.
calibration_names = []
calibration = pd.DataFrame()

# Code for multiple files upload option
if input_format == "Multiple files":
    # The user inputs the number of compounds for calibration
    compounds = st.number_input("Enter number of compounds for calibration:", step=1)
    # Initialize an empty list to hold the names of the compounds
    compound_names = []
    # Loop through the number of compounds, get the names from the user and append to the list

    for compound in range(compounds):
        compound_names.append(st.text_input(f"Enter compound {compound} name:"))

    # Upload multiple CSV files for calibration data
    calibration_files = st.file_uploader("Upload calibration data", type = "csv", accept_multiple_files = True)
    if calibration_files == None:
        # Stop execution if no file is uploaded
        st.stop()

    # Loop through each uploaded file
    for calibration_file in calibration_files:
        # Load each file into a DataFrame with specific column names
        calibration_sample = pd.read_csv(calibration_file, header=None, names=["Wavelength", calibration_file.name],
                                         # Remove any missing data (rows with NaNs)
                                         calibration_sample=calibration_sample.dropna()
        # Add the name of each file to a list
        calibration_names.append(calibration_file.name)
        # Concatenate the data from each file into a single DataFrame
        calibration = pd.concat([calibration, calibration_sample[calibration_file.name]], axis=1)

    # Create a DataFrame to hold calibration concentrations with columns for each compound and rows for each file
    calibration_concentrations = pd.DataFrame(columns=compound_names, index=range(len(calibration.columns)))

    # Insert a new column to hold file names
    column_names = calibration_concentrations.columns.tolist()
    column_names.insert(0, "File name")
    calibration_concentrations = calibration_concentrations.reindex(columns=column_names)

    # Populate the 'File name' column with the names of the uploaded files
    calibration_concentrations["File name"] = calibration_names

    # Configure data editor to accept float values
    column_dict = {}
    for compound_name in column_names:
        column_dict[compound_name] = {compound_name: st.column_config.NumberColumn(
            compound_name,
            min_value=0, format="%.3f",
        )}

        # Create data editor to allow the user to input calibration concentrations
        calibration_concentrations = st.data_editor(calibration_concentrations)

        # Display the filled in calibration concentrations DataFrame
        st.write(calibration_concentrations)
        # Set the column names of the calibration data DataFrame to match the 'File name' column from the concentrations DataFrame
        calibration.columns = calibration_concentrations["File name"]
        # Set the index of the concentrations DataFrame to the 'File name' column
        calibration_concentrations = calibration_concentrations.set_index("File name")

# Code for single file upload option
elif input_format == "Single file":
    # Upload a single CSV file for calibration data
    calibration_file = st.file_uploader("Upload calibration data", type="csv")
    # Stop execution if no file is uploaded
    if calibration_file is None:
        st.stop()

    # Load the file into a DataFrame, setting the first column as the index
    calibration_file = pd.read_csv(calibration_file, index_col=0)
    # Set the calibration data DataFrame to the loaded file data
    calibration = calibration_file
    # Display the loaded calibration data
    st.write(calibration)
    # Upload a single CSV file for calibration concentrations    calibration_conc = st.file_uploader("Upload calibration concentrations", type = "csv")
    if calibration_conc == None:
        st.stop()
    calibration_conc = pd.read_csv(calibration_conc, index_col = 0)
    calibration_conc = calibration_conc.reset_index(drop= True)
    compound_names = calibration_conc.columns.tolist()
    calibration_concentrations = calibration_conc
    st.write(calibration_conc)


# Introduction to the wavelength filtering section of the app
st.markdown("## 2. Filter data")
st.markdown("Data is frequently noisy on the edges of the spectrum. Therefore, we recommend removing this region prior to data analysis.")

# Let user decide how much data to process
# Define filter range
step_size = int(calibration.index[1] - calibration.index[0])

# The user inputs the number of wavelength ranges to consider
variable_selection = st.number_input("Select number of wavelength ranges of interest (Variable selection):",
                                     value = 1, step = 1)

# Initialize an empty list to hold filter range tuples
filter_range = []

# Loop through the number of variables, let the user choose a range for each, and append to the list
for variable in range(variable_selection):
    filter_range.append(st.slider(f"Select filter range {variable}", int(min(calibration.index)), int(max(calibration.index)),
                         (int(min(calibration.index)), int(max(calibration.index))), step = step_size))

# Make a copy of the full calibration data before applying the filter
full_calibration = calibration.copy()

# Apply the filter by keeping only the data within the selected ranges
mask = pd.Series(False, index=calibration.index, dtype=bool)
for start, end in filter_range:
    mask = mask | calibration.index.isin(range(start, end+1))

calibration = calibration[mask]

# Display a line plot of the filtered data
fig = px.line(calibration, x = calibration.index, y = calibration.columns[:], title = "Raw data")
st.plotly_chart(fig)

# Introduction to the Savitzky-Golay filter section of the app
st.markdown("## 3. Savitzky-Golay Filter")
st.markdown("Explanation of what each parameter does")

# Let the user decide whether to use the Savitzky-Golay filter and specify parameters
st.markdown("### Smoothing parameters")
smooth = st.checkbox("Use Savitzky-Golay filter?")
if smooth:
    col1, col2 = st.columns(2)

    with col1:
        window_size = st.slider("Window size", 1, 13, 9, step = 1)
    with col2:
        poly_order = st.slider("Polynomial order", 1, 5, 3, step = 1)

    # Apply the Savitzky-Golay filter and calculate the first and second derivatives
    calibration_d1 = pd.DataFrame(np.gradient(signal.savgol_filter(calibration, window_length=window_size,
                                                             polyorder=poly_order,axis=0), edge_order=2)[0], index = calibration.index, columns = calibration.columns)

    calibration_d2 = pd.DataFrame(np.gradient(signal.savgol_filter(calibration_d1, window_length=window_size,
                                                             polyorder=poly_order,axis=0), edge_order=2)[0], index = calibration.index, columns = calibration.columns)

else:
    # Calculate the first and second derivatives without the Savitzky-Golay filter
    calibration_d1 = pd.DataFrame(np.gradient(calibration, edge_order=2)[0], index = calibration.index, columns = calibration.columns)
    calibration_d2 = pd.DataFrame(np.gradient(calibration_d1, edge_order=2)[0], index = calibration.index, columns = calibration.columns)


# Display line plots of the raw, first derivative, and second derivative data
fig = px.line(calibration, x = calibration.index, y = calibration.columns[:], title = "Raw data")
st.plotly_chart(fig)
fig = px.line(calibration_d1, x = calibration_d1.index, y = calibration_d1.columns[:], title = "First derivative")
st.plotly_chart(fig)
fig = px.line(calibration_d2, x = calibration_d2.index, y = calibration_d2.columns[:], title = "Second derivative")
st.plotly_chart(fig)


# Begin section for user to set parameters for Partial Least Squares Regression (PLS)
st.markdown("## 4. Partial Least Squares Regression parameters")

col1, col2 = st.columns(2)
# Column 1: User chooses which format of data to analyze
with col1:
    st.markdown("Decide which data format do you want to analyse. Each format has its own advantages and disadvantages. "
                "The derivatives are recommended because they mitigate any shifts in baseline.")
    data_model = st.radio("Select with what do you want to make the analysis",
                          ["Intensity", "First derivative", "Second derivative"], index = 1)

# Column 2: User sets threshold for minimum variance explained by PCA - for visualisation purposes
with col2:
    st.markdown("When running a PCA you can use a lot more components than needed. "
                "This is a filter to avoid running PCA without physical meaning. If unsure, leave at zero. ")
    threshold = st.slider("Select the threshold for the minimum variance explained by the PCA.",
                          0.0, 0.2, 0.001, step = 0.0001, format = "%.4f")

# Select the correct data to analyze based on user choice
if data_model == "Intensity":
    data = calibration
elif data_model == "First derivative":
    data = calibration_d1
elif data_model == "Second derivative":
    data = calibration_d2

# Try to run PCA and find the optimal number of components that explain variance above the threshold
# For visualisation purposes
try:
    for i in range(1, len(data.columns)):
        pca = PCA(n_components=i)
        pca.fit(data.transpose())
        if pca.explained_variance_ratio_[-1] < threshold:
            break

    pca_results = pd.DataFrame(pca.explained_variance_ratio_,
                               columns = ["Variance explained"], index = range(1, len(pca.explained_variance_ratio_)+1))
# If PCA fails, fit it with the maximum number of components
except:
    pca = PCA(n_components=len(calibration.index))
    pca.fit(data.transpose())
    pca_results = pd.DataFrame(pca.explained_variance_ratio_,
                               columns = ["Variance explained"], index = range(1, len(pca.explained_variance_ratio_)+1))

# Plot the variance explained by each PCA component
fig = px.line(pca_results, title = "PCA explained variance ratio")
st.plotly_chart(fig)

# User selects the number of components to use in PLS
components = st.radio("Select number of components", range(1, len(pca_results)), index = 0)


# Plot PLS diagnostics: scores
full_data = data

# Start cross validation: leave one out (loo) approach is used
loo = LeaveOneOut()
loo.get_n_splits(full_data.T)

# Initialize DataFrames to hold the predicted and actual results
results = pd.DataFrame(columns = compound_names)
true_values = pd.DataFrame(columns = compound_names)

# Perform cross-validation, fitting the PLS model on the training set and making predictions on the test set
for i, (train_index, test_index) in enumerate (loo.split(full_data.T)):
    pls = PLSRegression(n_components=components)
    pls.fit(full_data.T.iloc[train_index].reset_index(drop=True),calibration_concentrations.iloc[train_index].reset_index(drop=True))
    result = pls.predict(full_data.T.iloc[test_index].reset_index(drop=True))
    true_value = calibration_concentrations.iloc[test_index].reset_index(drop=True)

    # Append the predictions and true values to the respective DataFrames
    true_values = pd.concat([true_values, true_value], axis = 0)
    results = pd.concat([results, pd.DataFrame(result, columns = compound_names)])


# Begin section to display results of the model
st.markdown("## 5. Results")
st.markdown("All metrics calculated using a Leave-One-Out Cross-validation")

# Initialize a DataFrame to hold results
results_df = pd.DataFrame(index = ["Global"] + compound_names)

# Define a dictionary of metrics to be calculated
metrics = {"R^2": R2, "Explained variance score": EVS,
           "Mean squared error": MSE, "Mean absolute error": MAE}

# Calculate each metric for each compound and for the global dataset
for metric in metrics.keys():
    score = []
    # Compute global score
    score.append(metrics[metric](true_values, results))
    # Compute score for each compound
    for i in range(len(compound_names)):
        score.append(metrics[metric](true_values.iloc[:,i], results.iloc[:,i]))
    # Add scores to results DataFrame
    results_df[metric] = score

# Display the results DataFrame in the Streamlit app
st.write(results_df)

# Create DataFrame to compare true and predicted values
true_vs_pred = pd.DataFrame(columns = ["True values", "Predicted values", "Compound"])

# Populate the DataFrame with the true and predicted values for each compound
for compound in compound_names:

    entry_tvpred = pd.DataFrame({"True values": list(true_values[compound]),
                                 "Predicted values": list(results[compound]),
                                 "Compound" :[compound]*len(true_values[compound])})
    true_vs_pred = pd.concat([true_vs_pred, entry_tvpred], axis = 0)

# Plot predicted vs actual values
fig = px.scatter(true_vs_pred, x = "True values", y = "Predicted values", color = "Compound")
try:
    for i in range(len(true_vs_pred["True values"])):
        # Convert values to floats
        true_vs_pred["True values"].iloc[i] = float(true_vs_pred["True values"].iloc[i])
        true_vs_pred["Predicted values"].iloc[i] = float(true_vs_pred["Predicted values"].iloc[i])
    # Add line to the plot
    fig.add_shape(
        type='line', line=dict(color = "white", width=2, dash="dash"),
        x0=min(min(true_vs_pred["True values"]), min(true_vs_pred["Predicted values"])),
        x1=max(max(true_vs_pred["True values"]), max(true_vs_pred["Predicted values"])),
        y0=min(min(true_vs_pred["True values"]), min(true_vs_pred["Predicted values"])),
        y1=max(max(true_vs_pred["True values"]), max(true_vs_pred["Predicted values"]))
    )
except:
    pass

# Display the plot in the Streamlit app
st.plotly_chart(fig)

# Fit the PLS model to all the data
pls = PLSRegression(n_components=components)
pls.fit(data.transpose().reset_index(drop=True),calibration_concentrations.reset_index(drop=True))

# Begin section for making predictions on sample data
st.markdown("## 5. Predictions")

# User uploads sample files for prediction
sample_files = st.file_uploader("Upload sample data", type = "csv", accept_multiple_files = True) #Header = wavelength followed by names

# Check if any files were uploaded
if len(sample_files) is 0:
    st.write("Please upload a sample file")

else:

    sample_data = pd.DataFrame()
    # Load and preprocess each sample file
    for sample_file in sample_files:

        sample = pd.read_csv(sample_file, header = None, names = ["Wavelength", sample_file.name], index_col = "Wavelength")
        sample = sample[mask]
        sample_data = pd.concat([sample_data, sample], axis = 1)

    if smooth:

        sample_d1 = pd.DataFrame(np.gradient(signal.savgol_filter(sample_data, window_length=window_size,
                                                                     polyorder=poly_order,axis=0), edge_order=2, axis = 0), index = sample_data.index, columns = sample_data.columns)

        sample_d2 = pd.DataFrame(np.gradient(signal.savgol_filter(sample_d1, window_length=window_size,
                                                                     polyorder=poly_order,axis=0), edge_order=2, axis = 0), index = sample_data.index, columns = sample_data.columns)
    else:
        sample_d1 = pd.DataFrame(np.gradient(sample_data, edge_order=2)[0], index = sample_data.index, columns = sample_data.columns)
        sample_d2 = pd.DataFrame(np.gradient(sample_d1, edge_order=2)[0], index = sample_data.index, columns = sample_data.columns)

    if data_model == "Intensity":
        sample_data = sample_data
    elif data_model == "First derivative":
        sample_data = sample_d1
    elif data_model == "Second derivative":
        sample_data = sample_d2


    predictions = pls.predict(pd.DataFrame(sample_data.iloc[:,:]).transpose())
    predictions = pd.DataFrame(predictions, columns = compound_names, index = sample_data.columns)

    st.write(predictions)

    st.download_button("Download predictions", data = predictions.to_csv(), file_name = "predictions_.csv")

    ## One solution to getting a confidence interval is to get a prediction for each model from the CV.
    ## The distribution of predictions should be a normal distribution and can give a confidence interval.
    ## How applicable is this? We will have 26+ models, so 26+ predictions for each sample. There should be a better way
    ## to get a confidence interval.


