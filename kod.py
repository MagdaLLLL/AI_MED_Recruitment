import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# Import the data from the CSV file
dane = pd.read_csv('/task_data.csv')

# Replace commas with periods and convert to numeric
for col in ["Heart width", "Lung width", "CTR - Cardiothoracic Ratio", "xx", "yy", "xy", "normalized_diff", "Inscribed circle radius", "Polygon Area Ratio", "Heart perimeter", "Heart area ", "Lung area"]:
    dane[col] = dane[col].astype(str).str.replace(',', '.', regex=False)
    dane[col] = pd.to_numeric(dane[col])

# Select the columns (features) that the model will use to learn - X
# Select the target column ("Cardiomegaly"), which represents the value the model aims to predict - y
X = dane[["Heart width", "Lung width", "CTR - Cardiothoracic Ratio", "xx", "yy", "xy", "normalized_diff", "Inscribed circle radius", "Polygon Area Ratio", "Heart perimeter", "Heart area ", "Lung area"]]
y = dane["Cardiomegaly"]

# Split the dataset into training (80%) and testing (20%) parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Nearest Neighbors (KNN) Classifier
# Define a Pipeline that bundles preprocessing (scaling) and the model
pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),     
    ("model", KNeighborsClassifier(   
        n_neighbors = 3,                
        weights='distance',           
        metric='manhattan'            
    ))
])

# Fit the Pipeline on the training data
pipe_knn.fit(X_train, y_train)

# Evaluate the model using cross-validation WITHOUT leakage
cv_score = np.round(cross_val_score(pipe_knn, X_train, y_train), 2)

# Display detailed results
print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}")
