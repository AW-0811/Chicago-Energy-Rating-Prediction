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



df = random_imputation(df, 'ZIP Code')
df = random_imputation(df, 'Community Area')
df = random_imputation(df, 'Primary Property Type')


# df = mean_imputation(df, 'Year Built')
# df = mean_imputation(df, '# of Buildings')

df = mean_imputation(df, 'Gross Floor Area - Buildings (sq ft)')
df = mean_imputation(df, 'Electricity Use (kBtu)')
# df = mean_imputation(df, 'Natural Gas Use (kBtu)')
df = mean_imputation(df, 'Weather Normalized Site EUI (kBtu/sq ft)')

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

imputer = KNNImputer(n_neighbors=7)

# Impute missing values
dfnotdf = imputer.fit_transform(df)
df = pd.DataFrame(dfnotdf, columns=df.columns)
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
X_train, X_test, y_train, y_test = train_test_split(Features, target, test_size=0.1, random_state=13)
# # LINEAR REGRESSION 
# regr = linear_model.LinearRegression()

# regr.fit(X_train, y_train)

# lin_reg_y_pred = regr.predict(X_test)

# lin_reg_y_pred_rounded = np.round(lin_reg_y_pred * 2) / 2

# mse = mean_absolute_error(y_test, lin_reg_y_pred_rounded)
# print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
# r2 = r2_score(y_test, lin_reg_y_pred_rounded)
# print("The r2 score on test set: {:.4f}".format(r2))

# # LASSO REGRESSION

# l_reg = Lasso(random_state = 9)  
# l_reg.fit(X_train, y_train)    
# l_reg_Y_pred = l_reg.predict(X_test)


# for i,n in enumerate(l_reg_Y_pred):
#     l_reg_Y_pred[i] = round(n*2)/2

# mse = mean_absolute_error(y_test, l_reg_Y_pred)
# print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
# r2 = r2_score(y_test, l_reg_Y_pred)
# print("The r2 score on test set: {:.4f}".format(r2))

# # RIDGE REGRESSION 

# r_reg = Ridge(solver = 'svd', random_state = 10)     
# r_reg.fit(X_train, y_train)   
# r_reg_Y_pred = r_reg.predict(X_test)  

# for i,n in enumerate(r_reg_Y_pred):
#     r_reg_Y_pred[i] = round(n*2)/2

# mse = mean_absolute_error(y_test, r_reg_Y_pred)
# print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
# r2 = r2_score(y_test, r_reg_Y_pred)
# print("The r2 score on test set: {:.4f}".format(r2))

# XGBOOST REGRESSION
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

print(y_test.shape)
params = {
    "n_estimators": 1500,
    "max_depth": 10,
    "min_samples_split": 5,
    "learning_rate": 0.03,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

pred = reg.predict(X_test)
print(pred)
for i,n in enumerate(pred):
    pred[i] = round(n*2)/2
    if pred[i] == 0.5:
        pred[i] = round(n)

print(pred)

mae = mean_absolute_error(y_test, pred)
print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))
r2 = r2_score(y_test, pred)
print("The r2 score on test set: {:.4f}".format(r2))


# # NN Approach
# model = tf.keras.Sequential([
#     layers.Input(shape=(9,)),
#     layers.Dense(128, activation='relu'),  # Increase the number of neurons
#     layers.BatchNormalization(),
#     layers.Dropout(0.2),
#     layers.Dense(640, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(1200, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(3000, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(1200, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(600, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(320, activation='relu'),
#     layers.Dense(1)
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
#               loss='mean_absolute_error')
# model.fit(X_train, y_train, epochs=100, batch_size=1024)
# NN_predictions = model.predict(X_test)
# NN_predictions_rounded = np.round(NN_predictions * 2) / 2
# mae = mean_absolute_error(y_test, NN_predictions_rounded)
# print("The mean absolute error on test set: {:.4f}".format(mae))
# r2 = r2_score(y_test, NN_predictions_rounded)
# print("The r2 score on test set: {:.4f}".format(r2))

# explainer = shap.Explainer(reg, X_train)

# # Calculate SHAP values for the test set
# shap_values = explainer(X_test, check_additivity=False)

# # Plot summary plot (global interpretation)
# shap.summary_plot(shap_values, X_test)