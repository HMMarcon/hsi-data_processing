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

# Give page description and short tutorial
st.markdown("# Hyperspectral calibration setup")

st.markdown("## Introduction")
st.markdown("This app is designed as a walkthrough to calibrate hyperspectral image data. Here is a general schema of the data processing steps:")
# st.image("https://i.imgur.com/1Z0ZQ2M.png", width = 500) - add image later
st.markdown("## 1. Load data")
st.markdown("Upload calibration data and sample data. Both should be a CSV-file with the first column containing the "
            "wavelengths and the remaining columns containing the samples (either calibration of unknown samples). "
            "It is important to remember that the CSV-file should have a header with the names of the samples. ")

# Load calibration data (df_calib) and sample data (df_sample)
# Tell the user how data should be formatted (header = wavelength followed by concentrations)
calibration_file = st.file_uploader("Upload calibration data", type = "csv")
sample_file = st.file_uploader("Upload sample data", type = "csv") #Header = wavelength followed by names

calibration = pd.read_csv(calibration_file)
index = calibration.columns[0]
calibration = calibration.set_index(index)
calibration = calibration.dropna()



calibration_concentrations = []
col1, col2 = st.columns(2)

for i in range(len(calibration.columns)):
    if i < len(calibration.columns)/2:
        with col1:
            value = st.number_input("Enter concentration of " + calibration.columns[i], 0.0, None)
    else:
        with col2:
            value = st.number_input("Enter concentration of " + calibration.columns[i], 0.0, None)

    calibration_concentrations.append(value)



st.markdown("## 2. Filter data")
st.markdown("Data is frequently noisy on the edges of the spectrum. Therefore, we recommend removing this region prior to data analysis.")

# Let user decide how much data to process
# Define filter range

filter_range = st.slider("Select filter range", int(min(calibration.index)), int(max(calibration.index)),
                         (int(min(calibration.index)+98), int(max(calibration.index))-198), step = 1)



calibration = calibration.loc[filter_range[0]:filter_range[1]]

# User decide Sav-Gol filter parameters
st.markdown("## 3. Savitzky-Golay Filter")
st.markdown("Explanation of what each parameter does")


# Select window size and polynomial order

st.markdown("### Smoothing parameters")
smooth = st.checkbox("Use Savitzky-Golay filter?")
if smooth:
    col1, col2 = st.columns(2)

    with col1:
        window_size = st.slider("Window size", 1, 13, 5, step = 1)
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
st.markdown("Explanation of what each parameter does")

data_model = st.radio("Select with what do you want to make the analysis", ["Intensity", "First derivative", "Second derivative"])

if data_model == "Intensity":
    data = calibration
elif data_model == "First derivative":
    data = calibration_d1
elif data_model == "Second derivative":
    data = calibration_d2

pca = PCA(n_components=len(data.columns))
pca.fit(data.transpose())
pca_results = pd.DataFrame(pca.explained_variance_ratio_, columns = ["Variance explained"], index = range(len(data.columns)))


fig = px.line(pca_results, title = "PCA explained variance ratio")
st.plotly_chart(fig)

components = st.radio("Select number of components", range(1, len(data.columns)))


# Plot PLS diagnostics: scores

pls = PLSRegression(n_components=components)
pls.fit(data.transpose(),calibration_concentrations)

# Add Predicted vs Observed plot
#
# fig = go.Figure()
#
# fig.add_trace(go.Scatter(x = [min(calibration_concentrations), max(calibration_concentrations)],
#               y = [min(calibration_concentrations), max(calibration_concentrations)]))
#
# fig.add_trace(go.Scatter(x = pls.predict(data.transpose()), y = calibration_concentrations))
#
#
# fig = px.scatter(x = pls.predict(data.transpose()), y = calibration_concentrations, title = "PLS scores")
# st.plotly_chart(fig)


# Generate predictions with sample data
st.markdown("## 5. Predictions")

if sample_file is None:
    st.write("Please upload a sample file")

if sample_file is not None:
    sample = pd.read_csv(sample_file)
    index = sample.columns[0]
    sample = sample.set_index(index)
    sample = sample.dropna()

    sample = sample.loc[filter_range[0]:filter_range[1]]

    if smooth:

        sample_d1 = pd.DataFrame(np.gradient(signal.savgol_filter(sample, window_length=window_size,
                                                                 polyorder=poly_order,axis=0), edge_order=2)[0], index = sample.index, columns = sample.columns)

        sample_d2 = pd.DataFrame(np.gradient(signal.savgol_filter(sample_d1, window_length=window_size,
                                                                 polyorder=poly_order,axis=0), edge_order=2)[0], index = sample.index, columns = sample.columns)
    else:
        sample_d1 = pd.DataFrame(np.gradient(sample, edge_order=2)[0], index = calibration.index, columns = calibration.columns)
        sample_d2 = pd.DataFrame(np.gradient(sample_d1, edge_order=2)[0], index = calibration.index, columns = calibration.columns)

    if data_model == "Intensity":
        sample = sample
    elif data_model == "First derivative":
        sample = sample_d1
    elif data_model == "Second derivative":
        sample = sample_d2

    predictions = pls.predict(sample.transpose())

    st.write(predictions)
