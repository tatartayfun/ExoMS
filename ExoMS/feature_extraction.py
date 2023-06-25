import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# specify the path to the Excel file containing mz_axis
#mz_axis_file = 'mz_axis.xlsx'
mz_axis_file = 'unique_masses.xlsx'

# read the mz_axis file and store it in a numpy array
mz_axis_df = pd.read_excel(mz_axis_file, header=1)
mz_axis = mz_axis_df.to_numpy()

# specify the path to the Excel file
#excel_file = 'aligned_spectra.xlsx'
excel_file = 'feature_map.xlsx'

# read the Excel file and store each sheet in a dictionary of data frames
dfs_dict = pd.read_excel(excel_file, sheet_name=None, header=1)

# create an empty dictionary to store the mass spectra for each patient
patients_dict = {}

# loop through each sheet in the dfs_dict and extract the mass spectra for each patient
for sheet_name, df in dfs_dict.items():
    
    # loop through each pair of columns and extract the mass spectra for each patient
    for i in range(0, len(df.columns), 1):
        rel_intensities = df.iloc[:, i].dropna().tolist()
    
        patient_name = f"{sheet_name}_patient_{i+1}"
        patients_dict[patient_name] = rel_intensities

# Create a pandas DataFrame from the dictionary of patient mass spectra
patients_df = pd.DataFrame.from_dict(patients_dict)

#RANDOM FOREST CLASSIFIER

# Split the data into training and testing sets
train_size = int(patients_df.shape[1] * 0.7)
train_patients = patients_df.iloc[:, :train_size]
print(train_patients)
test_patients = patients_df.iloc[:, train_size:]

# Fit a random forest classifier to the training data to select the best peaks for classification
rf = RandomForestClassifier(n_estimators=1000, random_state=220)
rf.fit(train_patients.T, list(range(train_size)))

# Get the feature importances and sort them in descending order
feat_importances = rf.feature_importances_
sorted_idx = np.argsort(feat_importances)[::-1]

# Select the top 50 features with the highest importances
top_feat_idx = sorted_idx[:50]

# create a DataFrame to store the mz_axis data
top_feat_idx_df = pd.DataFrame(top_feat_idx, columns=['peak_idx'])

# save the top_feat_idx_df DataFrame to an Excel file
with pd.ExcelWriter('top_peaks.xlsx') as writer:
    top_feat_idx_df.to_excel(writer, index=False)

#P VALUES
# Calculate the p-values for each peak using a t-test
p_values = []
for i in range(patients_df.shape[0]):
    group1 = train_patients.iloc[i, :]
    group2 = test_patients.iloc[i, :]
    t, p_value = ttest_ind(group1, group2, equal_var=False)
    p_values.append(p_value)
    
# create a DataFrame to store the mz_axis data
p_values_df = pd.DataFrame(p_values, columns=['p_value'])

# save the top_feat_idx_df DataFrame to an Excel file
with pd.ExcelWriter('p_values.xlsx') as writer:
    p_values_df.to_excel(writer, index=False)

# Convert the p-values to a pandas Series and sort them in ascending order
p_values_series = pd.Series(p_values)
sorted_p_values_series = p_values_series.sort_values()
print(sorted_p_values_series[:55])

# Select the top 50 peaks with the smallest p-values
top_feat_idx_pval = sorted_p_values_series.index[:10]

# Plot the feature importances
plt.bar(range(len(feat_importances)), np.sort(feat_importances)[::-1])
plt.xlabel('Feature index', fontsize=14)
plt.ylabel('Feature importance', fontsize=14)
plt.tick_params(axis='both', labelsize=12)  # Increase the font size of tick labels on both axes
plt.grid(True)  # Add grid to the plot
plt.show()
