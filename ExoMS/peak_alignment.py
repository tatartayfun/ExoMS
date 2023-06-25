import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from dtw import dtw
from scipy.spatial._kdtree import distance_matrix
from scipy.signal import find_peaks

# specify the path to the Excel file
excel_file = 'ac1c04762_si_002.xlsx'

# read the Excel file and store each sheet in a dictionary of data frames
dfs_dict = pd.read_excel(excel_file, sheet_name=None, header=1)

# create an empty dictionary to store the mass spectra for each patient
patients_dict = {}

# loop through each sheet in the dfs_dict and extract the mass spectra for each patient
for sheet_name, df in dfs_dict.items():
    # check if the number of columns in the sheet is even
    if len(df.columns) % 2 != 0:
        raise ValueError(f"Number of columns in sheet {sheet_name} is not even.")
    
    # loop through each pair of columns and extract the mass spectra for each patient
    for i in range(0, len(df.columns), 2):
        mz_ratios = df.iloc[:, i].dropna().tolist()
        rel_intensities = df.iloc[:, i+1].dropna().tolist()
    
        # sort the mz_ratios and rel_intensities in ascending order of mz_ratios
        mz_int_pairs = sorted(zip(mz_ratios, rel_intensities), key=lambda x: x[0])
        mz_ratios = [x[0] for x in mz_int_pairs]
        rel_intensities = [x[1] for x in mz_int_pairs]
    
        patient_name = f"{sheet_name}_patient_{(i//2)+1}"
        patients_dict[patient_name] = (mz_ratios, rel_intensities)
        
# extract the mass spectra from the patients_dict
mass_spectra = [patients_dict[key] for key in patients_dict.keys()]

# get the minimum and maximum m/z ratios across all spectra
min_mz = min([min(spectrum[0]) for spectrum in mass_spectra])
max_mz = max([max(spectrum[0]) for spectrum in mass_spectra])

# generate a list of m/z ratios to use as the x-axis for alignment
mz_axis = np.arange(min_mz, max_mz + 10, 10)

# create an empty array to store the aligned mass spectra
aligned_spectra = np.empty((len(mass_spectra), len(mz_axis)))

# create a dictionary to store the aligned spectra for each patient
aligned_dict = {}

# loop through each spectrum and align it to the mz_axis
for i, spectrum in enumerate(mass_spectra):
    # check that mz_ratios are sorted in ascending order
    mz_ratios, rel_intensities = spectrum[0], spectrum[1]
    if mz_ratios != sorted(mz_ratios):
        mz_ratios, rel_intensities = zip(*sorted(zip(mz_ratios, rel_intensities)))
        mass_spectra[i] = (mz_ratios, rel_intensities)
          
    # use the DTW algorithm to find the optimal path through the distance matrix
    dist, _, _, alignment_path = dtw(mz_axis.reshape(-1, 1), np.array(spectrum[0]).reshape(-1, 1), dist=euclidean)
    alignment_path = np.array(alignment_path)
    
    # interpolate the relative intensities of the spectrum onto the mz_axis using the alignment path
    aligned_intensities = np.interp(mz_axis, np.array(spectrum[0]), np.array(spectrum[1]))
    print(i)
    
    # normalize the intensities to the maximum intensity across all spectra
    #aligned_intensities = aligned_intensities / max(aligned_intensities)
    
    # store the aligned intensities in the aligned_spectra array
    aligned_spectra[i, :] = aligned_intensities
    
    # store the aligned intensities in the aligned_dict dictionary
    patient_name = f"patient_{i+1}"
    aligned_dict[patient_name] = aligned_intensities

# create a DataFrame to store the mz_axis data
mz_axis_df = pd.DataFrame(mz_axis, columns=['mz_axis'])

# save the mz_axis DataFrame to an Excel file
with pd.ExcelWriter('mz_axis.xlsx') as writer:
    mz_axis_df.to_excel(writer, index=False)
          
# save the aligned spectra to an Excel file
with pd.ExcelWriter('aligned_spectra.xlsx') as writer:
    for sheet_name, df in dfs_dict.items():
        # create an empty data frame to store the aligned spectra for this sheet
        aligned_df = pd.DataFrame()
        aligned_df["m/z ratios"]=mz_axis
        
        # loop through each pair of columns and extract the mass spectra for each patient
        for i in range(0, len(df.columns), 2):
            patient_name = f"patient_{(i//2)+1}"
            
            # check if the patient name exists in the aligned_dict
            if patient_name in aligned_dict:
                # add the aligned spectrum to the aligned_df data frame
                aligned_df[patient_name] = aligned_dict[patient_name]
            else:
                # add a column of NaN values to the aligned_df data frame
                aligned_df[patient_name] = np.nan
        
        # write the aligned_df data frame to the Excel file as a sheet with the same name as the original sheet
        aligned_df.to_excel(writer, sheet_name=sheet_name, index=False)