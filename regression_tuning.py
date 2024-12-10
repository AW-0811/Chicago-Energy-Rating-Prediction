import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def random_imputation(df, column):
    # Get the values of the column excluding NaN values
    non_null_values = df[column].dropna().values
    
    # Impute NaN values with random choices from non-null values
    df[column] = df[column].apply(lambda x: np.random.choice(non_null_values) if pd.isna(x) else x)
    
    return df

def mean_imputation(df, column):
    # Calculate the mean of the column, excluding NaN values
    mean_value = df[column].mean()
    
    # Impute NaN values with the calculated mean
    df[column] = df[column].fillna(mean_value)
    
    return df

df = pd.read_csv("Chicago_Energy_Benchmarking.csv")
unique, counts = np.unique(df['Chicago Energy Rating'], return_counts=True)
print (dict(zip(unique, counts)))
# print(df.head())
df = df.drop('Property Name', axis=1)
# print(df.head())

#main preprocessing starts here

df = df.dropna(subset=['Chicago Energy Rating'], axis=0)



# df = random_imputation(df, 'ZIP Code')
# df = random_imputation(df, 'Community Area')
# df = random_imputation(df, 'Primary Property Type')


# # df = mean_imputation(df, 'Year Built')
# # df = mean_imputation(df, '# of Buildings')

# df = mean_imputation(df, 'Gross Floor Area - Buildings (sq ft)')
# df = mean_imputation(df, 'Electricity Use (kBtu)')
# # df = mean_imputation(df, 'Natural Gas Use (kBtu)')
# df = mean_imputation(df, 'Weather Normalized Site EUI (kBtu/sq ft)')

df = df.drop('Water Use (kGal)', axis=1)
df = df.drop('District Steam Use (kBtu)', axis=1)
df = df.drop('District Chilled Water Use (kBtu)', axis=1)


# print(df.shape[0])

# missing_per_column = df.isnull().sum()

# # Count total missing values in the DataFrame
# total_missing = df.isnull().sum().sum()

# print("Missing values per column:\n", missing_per_column)
# print("Total missing values:", total_missing)

le = LabelEncoder()
df['ZIP_encoded'] = le.fit_transform(df['ZIP Code'])
df = df.drop('ZIP Code', axis=1)
# print(df.head())
df['Commarea_encoded'] = le.fit_transform(df['Community Area'])
df = df.drop('Community Area', axis=1)
df['Primary Property Type_encoded'] = le.fit_transform(df['Primary Property Type'])
df = df.drop('Primary Property Type', axis=1)
df = df.drop('Site EUI (kBtu/sq ft)', axis=1)
df = df.drop('Source EUI (kBtu/sq ft)', axis=1)
df = df.drop('Total GHG Emissions (Metric Tons CO2e)', axis=1)
df = df.drop('GHG Intensity (kg CO2e/sq ft)', axis=1)
df = df.drop('ENERGY STAR Score', axis=1)
# print(df.head())

df = df.drop('Weather Normalized Source EUI (kBtu/sq ft)', axis=1)

# imputer = KNNImputer(n_neighbors=7)

# # Impute missing values
# dfnotdf = imputer.fit_transform(df)
# df = pd.DataFrame(dfnotdf, columns=df.columns)
missing_per_column = df.isnull().sum()

# Count total missing values in the DataFrame
total_missing = df.isnull().sum().sum()

# print("Missing values per column:\n", missing_per_column)
# print("Total missing values:", total_missing)
# print(df.shape[0])
# sns.heatmap(df.corr(),vmin=-1, vmax=1, annot=True)


# print(plt.show())


Features = df.drop('Chicago Energy Rating', axis=1)
target = df['Chicago Energy Rating']
print(target)
X_train, X_temp, y_train, y_temp = train_test_split(Features, target, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

X_train = random_imputation(X_train, 'ZIP_encoded')
X_train = random_imputation(X_train, 'Commarea_encoded')
X_train = random_imputation(X_train, 'Primary Property Type_encoded')


# df = mean_imputation(df, 'Year Built')
# df = mean_imputation(df, '# of Buildings')

X_train = mean_imputation(X_train, 'Gross Floor Area - Buildings (sq ft)')
X_train = mean_imputation(X_train, 'Electricity Use (kBtu)')
# df = mean_imputation(df, 'Natural Gas Use (kBtu)')
X_train = mean_imputation(X_train, 'Weather Normalized Site EUI (kBtu/sq ft)')


imputer = KNNImputer(n_neighbors=7)

# Impute missing values
dfnotdf = imputer.fit_transform(X_train)
X_train = pd.DataFrame(dfnotdf, columns=X_train.columns)


X_val = random_imputation(X_val, 'ZIP_encoded')
X_val = random_imputation(X_val, 'Commarea_encoded')
X_val = random_imputation(X_val, 'Primary Property Type_encoded')


# df = mean_imputation(df, 'Year Built')
# df = mean_imputation(df, '# of Buildings')

X_val = mean_imputation(X_val, 'Gross Floor Area - Buildings (sq ft)')
X_val = mean_imputation(X_val, 'Electricity Use (kBtu)')
# df = mean_imputation(df, 'Natural Gas Use (kBtu)')
X_val = mean_imputation(X_val, 'Weather Normalized Site EUI (kBtu/sq ft)')


imputer = KNNImputer(n_neighbors=7)

# Impute missing values
dfnotdfvalid = imputer.fit_transform(X_val)
X_val = pd.DataFrame(dfnotdfvalid, columns=X_val.columns)

X_test = random_imputation(X_test, 'ZIP_encoded')
X_test = random_imputation(X_test, 'Commarea_encoded')
X_test = random_imputation(X_test, 'Primary Property Type_encoded')


# df = mean_imputation(df, 'Year Built')
# df = mean_imputation(df, '# of Buildings')

X_test = mean_imputation(X_test, 'Gross Floor Area - Buildings (sq ft)')
X_test = mean_imputation(X_test, 'Electricity Use (kBtu)')
# df = mean_imputation(df, 'Natural Gas Use (kBtu)')
X_test = mean_imputation(X_test, 'Weather Normalized Site EUI (kBtu/sq ft)')


imputer = KNNImputer(n_neighbors=7)

dfnotdftest = imputer.fit_transform(X_test)
X_test = pd.DataFrame(dfnotdftest, columns=X_test.columns)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

param_grid = {
    "n_estimators": [100, 200, 500, 1500],
    "max_depth": [3, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "learning_rate": [0.01, 0.03, 0.1, 0.2],
}
reg = GradientBoostingRegressor(loss="squared_error", random_state=42)

grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)

print("Starting hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print("Best parameters from GridSearchCV:", grid_search.best_params_)

best_model = grid_search.best_estimator_

print("Training best model...")
best_model.fit(X_train, y_train)

y_val_pred = best_model.predict(X_val)

mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
print(f"Validation MAE: {mae_val:.4f}")
print(f"Validation R²: {r2_val:.4f}")

y_test_pred = best_model.predict(X_test)
rounded_preds = [round(n*2)/2 for n in y_test_pred]

mae_test = mean_absolute_error(y_test, rounded_preds)
r2_test = r2_score(y_test, rounded_preds)
print(f"Test MAE: {mae_test:.4f}")
print(f"Test R²: {r2_test:.4f}")