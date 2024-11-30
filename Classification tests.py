import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Impute missing values in 'col1' using regression based on 'col2'
def regression_imputation(df, target_column, source_column):
    # Drop rows where either the target or the source column has missing values in the training data
    train_data = df.dropna(subset=[target_column, source_column])

    # If there are no valid training samples left, return the original dataframe
    if train_data.empty:
        return df

    # Define the regression model
    model = LinearRegression()

    # Train the model using valid rows only (rows where both columns are not missing)
    model.fit(train_data[[source_column]], train_data[target_column])

    # Predict missing values in the target column
    missing_data = df[df[target_column].isna()]

    # Loop through the rows with missing target values
    for index, row in missing_data.iterrows():
        # Only predict for rows where the feature is not missing
        if not pd.isna(row[source_column]):
            predicted_value = model.predict([[row[source_column]]])[0]
            df.loc[index, target_column] = predicted_value
        else:
            # Leave the target value as NaN if the feature is also NaN
            df.loc[index, target_column] = np.nan

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

df = mean_imputation(df, 'Year Built')
df = mean_imputation(df, '# of Buildings')

df = mean_imputation(df, 'Gross Floor Area - Buildings (sq ft)')
df = mean_imputation(df, 'Electricity Use (kBtu)')
df = mean_imputation(df, 'Natural Gas Use (kBtu)')
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

print("Unique values in y before fixing:", np.unique(target))

# Fix y to have integer class labels
_, target = np.unique(target, return_inverse=True)

# Verify fixed y
print("Unique values in y after fixing:", np.unique(target))

unique, counts = np.unique(target, return_counts=True)
print (dict(zip(unique, counts)))
# # y_one_hot = tf.keras.utils.to_categorical(target, num_classes=8)
# # X_trainval, X_test, y_trainval, y_test = train_test_split(Features, y_one_hot, test_size=0.1, random_state=13, stratify=y_one_hot)
# # X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval)
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_val = scaler.fit_transform(X_val)
# # X_test = scaler.fit_transform(X_test)

X_trainval, X_test, y_trainval, y_test = train_test_split(Features, target, test_size=0.1, random_state=13, stratify=target)
scaler = StandardScaler()
X_trainval = scaler.fit_transform(X_trainval)
X_test = scaler.fit_transform(X_test)

# # model = tf.keras.Sequential([
# #     layers.Input(shape=(9,)),
# #     layers.Dense(128, activation='relu'),  # Increase the number of neurons
# #     layers.Dense(640, activation='relu'),
# #     layers.Dense(320, activation='relu'),
# #     layers.Dense(8, activation='softmax')
# # ])
# # # Compile the model
# # model.compile(optimizer='adam',
# #               loss='categorical_crossentropy',
# #               metrics=['accuracy'])

# # # Train the model
# # history = model.fit(X_train, y_train,
# #                     validation_data=(X_val, y_val),
# #                     epochs=1000,
# #                     batch_size=512)

# # # Evaluate the model
# # loss, accuracy = model.evaluate(X_test, y_test)
# # print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


# # xgb_model = XGBClassifier(
# #     objective='multi:softmax',  # For multi-class classification
# #     num_class=8,               # Number of classes   # Avoid deprecated warnings
# #     eval_metric='mlogloss'     # Multi-class log loss
# # )

# # # Fit the model
# # xgb_model.fit(X_trainval, y_trainval)

# # # Make predictions
# # y_pred = xgb_model.predict(X_test)

# # # Evaluate the model
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Test Accuracy: {accuracy:.4f}")

# # # Classification report
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))

# # knn_model = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors

# # # Fit the model
# # knn_model.fit(X_trainval, y_trainval)

# # # Make predictions
# # y_pred = knn_model.predict(X_test)

# # # Evaluate the model
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Test Accuracy: {accuracy:.4f}")

# # # Classification report
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))


# # from sklearn.naive_bayes import GaussianNB

# # # Define the model
# # nb_model = GaussianNB()

# # # Fit and evaluate the model
# # nb_model.fit(X_trainval, y_trainval)
# # y_pred = nb_model.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)

# # print(f"Test Accuracy: {accuracy:.4f}")

# # # Classification report
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))

# # from sklearn.linear_model import LogisticRegression

# # # Define the model
# # lr_model = LogisticRegression( solver='lbfgs', max_iter=10000)

# # # Fit and evaluate the model
# # lr_model.fit(X_trainval, y_trainval)
# # y_pred = lr_model.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)

# # print(f"Test Accuracy: {accuracy:.4f}")

# # # Classification report
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))

# from sklearn.svm import SVC

# # Define the model
# svm_model = SVC(decision_function_shape='ovo')  # One-vs-one strategy

# # Fit and evaluate the model
# svm_model.fit(X_trainval, y_trainval)
# y_pred = svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Test Accuracy: {accuracy:.4f}")

# # Classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))


print(y_test.shape)
params = {
    "n_estimators": 1000,
    "max_depth": 10,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

