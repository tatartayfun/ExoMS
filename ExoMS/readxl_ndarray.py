
import pandas as pd
import numpy as np

excel_filename = "ac1c04762_si_002.xlsx"

def raw_sheet_data(dataframe):
    
    return_this = dataframe[1 : ][:].copy()
    
    num_of_col = return_this.shape[1]
    
    num_of_patients = num_of_col / 2
    
    assert num_of_patients.is_integer(), "Number of columns should be even number!"
    
    return_this.columns = range(int(num_of_col))
    
    return return_this
    

def convert_sheet_into_patient_list(dataframe):
    
    patients = []
    
    raw_dataframe = raw_sheet_data(dataframe)
    
    num_of_patients = int(raw_dataframe.shape[1] / 2)
    
    for idx in range(num_of_patients):
        masses = raw_dataframe[:][idx * 2].dropna(axis=0)
        intensities = raw_dataframe[:][idx * 2 + 1].dropna(axis=0)
        
        assert len(masses) == len(intensities), "Number of masses and intentisies for the patient is not equal!"
        
        vstacked_masses_and_intensities = pd.concat([masses, intensities], axis=1)
        patients.append(vstacked_masses_and_intensities)
        
    return patients

def convert_3_sheets_into_masses_and_intentisies(excel_name):
    
    df_1 = pd.read_excel(excel_name, sheet_name=0)
    df_2 = pd.read_excel(excel_name, sheet_name=1)
    df_3 = pd.read_excel(excel_name, sheet_name=2)
    
    group_1 = convert_sheet_into_patient_list(df_1)
    group_2 = convert_sheet_into_patient_list(df_2)
    group_3 = convert_sheet_into_patient_list(df_3)
    
    all_patients = []
    
    for idx_patients in range(len(group_1)):
        
        all_patients.append(group_1[idx_patients])
        
    for idx_patients in range(len(group_2)):
        
        all_patients.append(group_2[idx_patients])
        
    for idx_patients in range(len(group_3)):
        
        all_patients.append(group_3[idx_patients])
        
    masses_and_intensities = np.zeros((1, 2))
    for each_patient in all_patients:
        masses_and_intensities = np.vstack((masses_and_intensities, each_patient))
        
    masses_and_intensities = masses_and_intensities[1 : ]
    
    masses_and_intensities = masses_and_intensities[np.argsort(masses_and_intensities[:, 0])]
    
    return masses_and_intensities

m_and_i = convert_3_sheets_into_masses_and_intentisies(excel_filename)

np.save("masses_and_intensities", m_and_i)

df_m_and_i = pd.DataFrame(m_and_i)
pd.DataFrame.to_csv(df_m_and_i, "masses_and_intensities.csv")

# m_and_i_2 = np.load("masses_and_intensities.npy", allow_pickle=True)






