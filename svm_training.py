# svm_training.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib

# učitamo HOG značajke i labele
data_path = 'hog_features_labels.joblib'
X, y = joblib.load(data_path)

# Provjera da dataset ima makar dvije klase
unique_labels = np.unique(y)
print(f"Unique labels: {unique_labels}")

if len(unique_labels) < 2:
    raise ValueError("Dataset contains less than 2 classes. Ensure labels are correctly assigned.")

# Podjela na test i train set
# 80% train i 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# treniranje SVM modela
svm = LinearSVC()
svm.fit(X_train, y_train)

# Procijena accuracy
accuracy = svm.score(X_test, y_test)
print(f'SVM Classifier Accuracy: {accuracy}')

# Potrebno je sačuvati model <3
model_path = '../svm_model.joblib'
joblib.dump(svm, model_path)
print(f'Model saved to {model_path}')