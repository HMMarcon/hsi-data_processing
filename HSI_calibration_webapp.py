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

compounds = st.number_input("Enter number of compounds for calibration:", step = 1)
compound_names = []
for compound in range(compounds):
    compound_names.append(st.text_input(f"Enter compound {compound} name:"))

# Load calibration data (df_calib) and sample data (df_sample)
# Tell the user how data should be formatted (header = wavelength followed by concentrations)
calibration_files = st.file_uploader("Upload calibration data", type = "csv", accept_multiple_files = True)

calibration_names = []
calibration = pd.DataFrame()

for calibration_file in calibration_files:
    calibration_sample = pd.read_csv(calibration_file, header = None, names = ["Wavelength", calibration_file.name], index_col = "Wavelength")

    calibration_sample = calibration_sample.dropna()
    calibration_names.append(calibration_file.name)
    calibration = pd.concat([calibration, calibration_sample[calibration_file.name]], axis = 1)

calibration_concentrations = pd.DataFrame(columns = compound_names, index = range(len(calibration.columns)))
column_names = calibration_concentrations.columns.tolist()
column_names.insert(0, "File name")
calibration_concentrations = calibration_concentrations.reindex(columns = column_names)
calibration_concentrations["File name"] = calibration_names



calibration_concentrations = st.experimental_data_editor(calibration_concentrations)


calibration.columns = calibration_concentrations["File name"]



calibration_concentrations = calibration_concentrations.set_index("File name")




st.markdown("## 2. Filter data")
st.markdown("Data is frequently noisy on the edges of the spectrum. Therefore, we recommend removing this region prior to data analysis.")

# Let user decide how much data to process
# Define filter range
step_size = calibration.index[1] - calibration.index[0]

filter_range = st.slider("Select filter range", int(min(calibration.index)), int(max(calibration.index)),
                         (int(min(calibration.index)), int(max(calibration.index))), step = 4)



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
pca_results = pd.DataFrame(pca.explained_variance_ratio_, columns = ["Variance explained"], index = range(1, len(data.columns)+1))


fig = px.line(pca_results, title = "PCA explained variance ratio")
st.plotly_chart(fig)

components = st.radio("Select number of components", range(1, len(data.columns)))


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
st.markdown(f"Overall R^2 value: {r2}")
for i in range(len(compound_names)):
    st.markdown(f"R^2 value for {compound_names[i]}: {R2(true_values.iloc[:,i], results.iloc[:,i])}")
    #st.image(plt.plot(list(true_values.iloc[:,i]), list(results.iloc[:,i])))
#st.write(true_values, results)
#plt.plot(true_values, results, 'o')


#Regression with all data

pls = PLSRegression(n_components=components)
pls.fit(data.transpose().reset_index(drop=True),calibration_concentrations.reset_index(drop=True))

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

sample_files = st.file_uploader("Upload sample data", type = "csv", accept_multiple_files = True) #Header = wavelength followed by names

if len(sample_files) is 0:
    st.write("Please upload a sample file")

else:

    sample_data = pd.DataFrame()

    for sample_file in sample_files:

        sample = pd.read_csv(sample_file, header = None, names = ["Wavelength", sample_file.name], index_col = "Wavelength")
        sample = sample.loc[filter_range[0]:filter_range[1]]
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
