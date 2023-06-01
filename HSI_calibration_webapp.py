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
# Tell the user how data should be formatted (header = wavelength followed by concentrations)
input_format = st.radio("Select data format", ("Single file", "Multiple files"))
calibration_names = []
calibration = pd.DataFrame()

if input_format == "Multiple files":
    compounds = st.number_input("Enter number of compounds for calibration:", step=1)
    compound_names = []
    for compound in range(compounds):
        compound_names.append(st.text_input(f"Enter compound {compound} name:"))

    calibration_files = st.file_uploader("Upload calibration data", type = "csv", accept_multiple_files = True)
    for calibration_file in calibration_files:
        calibration_sample = pd.read_csv(calibration_file, header=None, names=["Wavelength", calibration_file.name],
                                         index_col="Wavelength")

        calibration_sample = calibration_sample.dropna()
        calibration_names.append(calibration_file.name)
        calibration = pd.concat([calibration, calibration_sample[calibration_file.name]], axis=1)

    calibration_concentrations = pd.DataFrame(columns=compound_names, index=range(len(calibration.columns)))
    column_names = calibration_concentrations.columns.tolist()
    column_names.insert(0, "File name")
    calibration_concentrations = calibration_concentrations.reindex(columns=column_names)
    calibration_concentrations["File name"] = calibration_names

    calibration_concentrations = st.experimental_data_editor(calibration_concentrations)
    calibration.columns = calibration_concentrations["File name"]
    calibration_concentrations = calibration_concentrations.set_index("File name")

elif input_format == "Single file":
    calibration_file = st.file_uploader("Upload calibration data", type = "csv")
    if calibration_file == None:
        st.stop()

    calibration_file = pd.read_csv(calibration_file, index_col = 0)
    calibration = calibration_file
    st.write(calibration)
    calibration_conc = st.file_uploader("Upload calibration concentrations", type = "csv")
    if calibration_conc == None:
        st.stop()
    calibration_conc = pd.read_csv(calibration_conc, index_col = 0)
    calibration_conc = calibration_conc.reset_index(drop= True)
    compound_names = calibration_conc.columns.tolist()
    calibration_concentrations = calibration_conc
    st.write(calibration_conc)


st.markdown("## 2. Filter data")
st.markdown("Data is frequently noisy on the edges of the spectrum. Therefore, we recommend removing this region prior to data analysis.")

# Let user decide how much data to process
# Define filter range
step_size = int(calibration.index[1] - calibration.index[0])

variable_selection = st.number_input("Select number of wavelength ranges of interest (Variable selection):",
                                     value = 1, step = 1)

filter_range = []

for variable in range(variable_selection):
    filter_range.append(st.slider(f"Select filter range {variable}", int(min(calibration.index)), int(max(calibration.index)),
                         (int(min(calibration.index)), int(max(calibration.index))), step = step_size))

#filter_range = st.slider("Select filter range", int(min(calibration.index)), int(max(calibration.index)),
#                         (int(min(calibration.index)), int(max(calibration.index))), step = 4)



full_calibration = calibration.copy()

#calibration = calibration.loc[filter_range[0][0]:filter_range[0][1]]

mask = pd.Series(False, index=calibration.index, dtype=bool)
for start, end in filter_range:
    mask = mask | calibration.index.isin(range(start, end+1))

calibration = calibration[mask]


fig = px.line(calibration, x = calibration.index, y = calibration.columns[:], title = "Raw data")
st.plotly_chart(fig)

# User decide Sav-Gol filter parameters
st.markdown("## 3. Savitzky-Golay Filter")
st.markdown("Explanation of what each parameter does")


# Select window size and polynomial order

st.markdown("### Smoothing parameters")
smooth = st.checkbox("Use Savitzky-Golay filter?")
if smooth:
    col1, col2 = st.columns(2)

    with col1:
        window_size = st.slider("Window size", 1, 13, 9, step = 1)
    with col2:
        poly_order = st.slider("Polynomial order", 1, 5, 3, step = 1)

    calibration_d1 = pd.DataFrame(np.gradient(signal.savgol_filter(calibration, window_length=window_size,
                                                             polyorder=poly_order,axis=0), edge_order=2)[0], index = calibration.index, columns = calibration.columns)

    calibration_d2 = pd.DataFrame(np.gradient(signal.savgol_filter(calibration_d1, window_length=window_size,
                                                             polyorder=poly_order,axis=0), edge_order=2)[0], index = calibration.index, columns = calibration.columns)

else:
    calibration_d1 = pd.DataFrame(np.gradient(calibration, edge_order=2)[0], index = calibration.index, columns = calibration.columns)
    calibration_d2 = pd.DataFrame(np.gradient(calibration_d1, edge_order=2)[0], index = calibration.index, columns = calibration.columns)


# Plot filtered data and derivatives

fig = px.line(calibration, x = calibration.index, y = calibration.columns[:], title = "Raw data")
st.plotly_chart(fig)
fig = px.line(calibration_d1, x = calibration_d1.index, y = calibration_d1.columns[:], title = "First derivative")
st.plotly_chart(fig)
fig = px.line(calibration_d2, x = calibration_d2.index, y = calibration_d2.columns[:], title = "Second derivative")
st.plotly_chart(fig)


# User decide PLS parameters
# Select number of components
st.markdown("## 4. Partial Least Squares Regression parameters")

col1, col2 = st.columns(2)
with col1:
    st.markdown("Decide which data format do you want to analyse. Each format has its own advantages and disadvantages. "
                "The derivatives are recommended because they mitigate any shifts in baseline.")
    data_model = st.radio("Select with what do you want to make the analysis",
                          ["Intensity", "First derivative", "Second derivative"], index = 1)

with col2:
    st.markdown("When running a PCA you can use a lot more components than needed. "
                "This is a filter to avoid running PCA without physical meaning. If unsure, leave at zero. ")
    threshold = st.slider("Select the threshold for the minimum variance explained by the PCA.",
                          0.0, 0.2, 0.001, step = 0.0001, format = "%.4f")

if data_model == "Intensity":
    data = calibration
elif data_model == "First derivative":
    data = calibration_d1
elif data_model == "Second derivative":
    data = calibration_d2



try:
    for i in range(1, len(data.columns)):
        pca = PCA(n_components=i)
        pca.fit(data.transpose())
        if pca.explained_variance_ratio_[-1] < threshold:
            break

    pca_results = pd.DataFrame(pca.explained_variance_ratio_,
                               columns = ["Variance explained"], index = range(1, len(pca.explained_variance_ratio_)+1))
except:
    pca = PCA(n_components=len(calibration.index))
    pca.fit(data.transpose())
    pca_results = pd.DataFrame(pca.explained_variance_ratio_,
                               columns = ["Variance explained"], index = range(1, len(pca.explained_variance_ratio_)+1))


fig = px.line(pca_results, title = "PCA explained variance ratio")
st.plotly_chart(fig)

components = st.radio("Select number of components", range(1, len(pca_results)), index = 0)


# Plot PLS diagnostics: scores
full_data = data

#Leave one out cross validation

loo = LeaveOneOut()
loo.get_n_splits(full_data.T)

results = pd.DataFrame(columns = compound_names)
true_values = pd.DataFrame(columns = compound_names)

for i, (train_index, test_index) in enumerate (loo.split(full_data.T)):
    pls = PLSRegression(n_components=components)
    pls.fit(full_data.T.iloc[train_index].reset_index(drop=True),calibration_concentrations.iloc[train_index].reset_index(drop=True))
    result = pls.predict(full_data.T.iloc[test_index].reset_index(drop=True))
    true_value = calibration_concentrations.iloc[test_index].reset_index(drop=True)
    true_values = pd.concat([true_values, true_value], axis = 0)
    results = pd.concat([results, pd.DataFrame(result, columns = compound_names)])



r2 = R2(true_values, results)
exp_var_score = EVS(true_values, results)
mse = MSE(true_values, results)
mae = MAE(true_values, results)





st.markdown("## 5. Results")
st.markdown("All metrics calculated using a Leave-One-Out Cross-validation")

results_df = pd.DataFrame(index = ["Global"] + compound_names)
metrics = {"R^2": R2, "Explained variance score": EVS,
           "Mean squared error": MSE, "Mean absolute error": MAE}

for metric in metrics.keys():
    score = []
    score.append(metrics[metric](true_values, results))
    for i in range(len(compound_names)):
        score.append(metrics[metric](true_values.iloc[:,i], results.iloc[:,i]))
    results_df[metric] = score

st.write(results_df)

true_vs_pred = pd.DataFrame(columns = ["True values", "Predicted values", "Compound"])
for compound in compound_names:

    entry_tvpred = pd.DataFrame({"True values": list(true_values[compound]),
                                 "Predicted values": list(results[compound]),
                                 "Compound" :[compound]*len(true_values[compound])})
    true_vs_pred = pd.concat([true_vs_pred, entry_tvpred], axis = 0)


## Predicted vs Actual plot
fig = px.scatter(true_vs_pred, x = "True values", y = "Predicted values", color = "Compound")
fig.add_shape(
    type='line', line=dict(color = "white", width=2, dash="dash"),
    x0=min(true_vs_pred["True values"]), x1=max(true_vs_pred["True values"]),
    y0=min(true_vs_pred["Predicted values"]), y1=max(true_vs_pred["Predicted values"])
)
st.plotly_chart(fig)


#Regression with all data

pls = PLSRegression(n_components=components)
pls.fit(data.transpose().reset_index(drop=True),calibration_concentrations.reset_index(drop=True))



# Generate predictions with sample data
st.markdown("## 5. Predictions")

sample_files = st.file_uploader("Upload sample data", type = "csv", accept_multiple_files = True) #Header = wavelength followed by names

if len(sample_files) is 0:
    st.write("Please upload a sample file")

else:

    sample_data = pd.DataFrame()

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

    #st.write(pd.DataFrame(sample_data.iloc[:,0]).transpose())

    predictions = pls.predict(pd.DataFrame(sample_data.iloc[:,:]).transpose())
    predictions = pd.DataFrame(predictions, columns = compound_names, index = sample_data.columns)

    st.write(predictions)

    st.download_button("Download predictions", data = predictions.to_csv(), file_name = "predictions_.csv")


