

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

disease = pd.read_csv(r"HeartDisease/heart_disease.csv")
disease.head()
disease.shape
disease.info()

#Age
#Blood Pressure
#Cholesterol Level
#Exercise Habits
#Smoking
#High Blood Pressure
#Low HDL Cholesterol
#Alcohol Consumption
#Sleep Hours
#Heart Disease Status

disease = disease[['Age','Blood Pressure','Cholesterol Level','Exercise Habits','Smoking','High Blood Pressure','Low HDL Cholesterol','Alcohol Consumption','Sleep Hours','Heart Disease Status']]

disease.isnull().sum()

disease.dropna(inplace=True)
disease.duplicated().sum()
disease.describe()
disease['Heart Disease Status'].value_counts()



for col in disease.select_dtypes(include=['object']).columns:
    disease[col], _ = pd.factorize(disease[col])
imputer = SimpleImputer(strategy='mean')
disease.iloc[:, :] = imputer.fit_transform(disease)

X = disease.drop(columns=['Heart Disease Status'])
y = disease['Heart Disease Status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.shape, X_train.shape, X_test.shape)

RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
RF_model.fit(X_train, y_train)

LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(X_train, y_train)

y_pred_rf = RF_model.predict(X_test)
y_pred_lr = LR_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy:.2f}")

accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")

# print(classification_report(y_test, y_pred))
joblib.dump(RF_model, "random_forest_model.pkl")
joblib.dump(LR_model, "logistic_regression_model.pkl")

plt.figure(figsize=(10, 6))
sns.heatmap(disease.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=disease['Blood Pressure'], y=disease['Cholesterol Level'], hue=disease['Heart Disease Status'], palette='coolwarm')
plt.title("Blood Pressure vs Cholesterol")
plt.xlabel("Blood Pressure")
plt.ylabel("Cholesterol Level")
plt.show()


