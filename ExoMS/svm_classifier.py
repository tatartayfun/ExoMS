#token: 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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from skopt import BayesSearchCV

# specify the path to the Excel file containing mz_axis
peaks_file = 'top_peaks.xlsx'

# read the mz_axis file and store it in a numpy array
top_peaks_df = pd.read_excel(peaks_file, header=0)
top_feat_idx = top_peaks_df.to_numpy()
top_feat_idx = top_feat_idx[:50]

# specify the path to the Excel file containing mz_axis
mz_axis_file = 'unique_masses.xlsx'

# read the mz_axis file and store it in a numpy array
mz_axis_df = pd.read_excel(mz_axis_file, header=1)
mz_axis = mz_axis_df.to_numpy()

# specify the path to the Excel file
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

# separate the dataframe into three dataframes based on the group specified in the column name
BCP_df = patients_df.filter(regex='^BCPs')
PCP_df = patients_df.filter(regex='^PCPs')
HC_df = patients_df.filter(regex='^HCs')

# select only specific rows based on top_feat_idx
BCP_df_sel = BCP_df.iloc[np.transpose(top_feat_idx)[0]]
PCP_df_sel = PCP_df.iloc[np.transpose(top_feat_idx)[0]]
HC_df_sel = HC_df.iloc[np.transpose(top_feat_idx)[0]]

# merge the selected dataframes and create a label array
print(type(BCP_df_sel))
X = pd.concat([BCP_df_sel, PCP_df_sel, HC_df_sel], axis=1).to_numpy().T
y = np.hstack([np.ones(len(BCP_df_sel.columns)), np.full(len(PCP_df_sel.columns), 2), np.full(len(HC_df_sel.columns), 3)])

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=220, stratify=y)

"""
# define the parameter search space for the SVM classifier
param_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'n_estimators': (100, 1000),
    'max_depth': (1, 10),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
    'max_features': (0.1, 1.0, 'uniform')
}

# perform Bayesian optimization
opt = BayesSearchCV(
    GradientBoostingClassifier(),
    param_space,
    n_iter=50,  # number of optimization iterations
    scoring='accuracy',  # optimize for accuracy
    n_jobs=-1,  # number of CPU cores to use (-1 means all available cores)
    cv=5  # number of cross-validation folds
)
opt.fit(X_train, y_train)

# print the best parameters and score found during optimization
print("Best parameters found:")
print(opt.best_params_)
print("Best accuracy found:")
print(opt.best_score_)

# get the optimized classifier
mcc = opt.best_estimator_
"""
# train the model with selected classifier
select_classifier = 'GradientBoostingClassifier'
# select the optimizer based on the value of the select_optimizer variable
if select_classifier == 'SGDClassifier':
    mcc = SGDClassifier(loss='hinge', alpha=0.001, max_iter=1000, tol=1e-3, random_state=220)
elif select_classifier == 'LogisticRegression':
    mcc = LogisticRegression(penalty='l2', C=0.95, solver='sag', max_iter=1000, random_state=220)
elif select_classifier == 'DecisionTreeClassifier':
    mcc = DecisionTreeClassifier(random_state=220)
elif select_classifier == 'RandomForestClassifier':
    mcc = RandomForestClassifier(n_estimators=100, random_state=220)
elif select_classifier == 'GradientBoostingClassifier':
    mcc = GradientBoostingClassifier(learning_rate=0.2, random_state=220, verbose=2)
elif select_classifier == 'KNeighborsClassifier':
    mcc = KNeighborsClassifier(n_neighbors=3)
elif select_classifier == 'GaussianNB':
    mcc = GaussianNB()
elif select_classifier == 'SVC':
    mcc = SVC(kernel='linear', C=0.1023434889134749, gamma=0.0016522387663209827, random_state=220)
else:
    print(f"{select_classifier} is not a valid optimizer.")
    exit()
    
mcc.fit(X_train, y_train)

# make predictions on the test set and print the accuracy
y_pred = mcc.predict(X_test)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print(f'Test recall rate: {recall}')
print(f'Test precision: {precision}')
print(f'Test accuracy: {accuracy}')

# calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot the confusion matrix as a heatmap using seaborn
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', annot_kws={"fontsize": 18})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.show()

