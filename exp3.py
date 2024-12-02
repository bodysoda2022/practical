import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the training dataset
df = pd.read_csv("https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp2/2a.%20loan-train.csv")
# Convert categorical variables to dummy variables
df = pd.get_dummies(df)

# Drop unnecessary columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis=1)

# Rename columns for better readability
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed', 
       'Loan_Status_Y': 'Loan_Status'}
df.rename(columns=new, inplace=True)

# Define features (X) and target (y)
X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]

# Handle class imbalance using SMOTE
X, y = SMOTE().fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic Regression:
LRclassifier = LogisticRegression(solver='saga', max_iter=500, random_state=1)
LRclassifier.fit(X_train, y_train)
y_pred = LRclassifier.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
LRAcc = accuracy_score(y_pred, y_test)
print('Logistic Regression Accuracy: {:.2f}%'.format(LRAcc * 100))
LRcv_scores = cross_val_score(LRclassifier, X_train, y_train, cv=5)
print("Logistic Regression CV Scores:", LRcv_scores)

# Support Vector Classifier (SVC):
SVCclassifier = SVC(kernel='rbf', max_iter=500)
SVCclassifier.fit(X_train, y_train)
y_pred = SVCclassifier.predict(X_test)
print("\nSVC Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
SVCAcc = accuracy_score(y_pred, y_test)
print('SVC Accuracy: {:.2f}%'.format(SVCAcc * 100))
SVCcv_scores = cross_val_score(SVCclassifier, X_train, y_train, cv=5)
print("SVC CV Scores:", SVCcv_scores)

# Decision Tree:
scoreListDT = []
for i in range(2, 21):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))

plt.plot(range(2, 21), scoreListDT)
plt.xticks(np.arange(2, 21, 1))
plt.xlabel("Leaf Nodes")
plt.ylabel("Accuracy Score")
plt.title("Decision Tree Accuracy for Different Leaf Node Sizes")
plt.show()

DTAcc = max(scoreListDT)
print("Decision Tree Accuracy: {:.2f}%".format(DTAcc * 100))
DTcv_scores = cross_val_score(DTclassifier, X_train, y_train, cv=5)
print("Decision Tree CV Scores:", DTcv_scores)

# Random Forest:
scoreListRF = []
for i in range(2, 25):
    RFclassifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))

plt.plot(range(2, 25), scoreListRF)
plt.xticks(np.arange(2, 25, 1))
plt.xlabel("Max Leaf Nodes")
plt.ylabel("Accuracy Score")
plt.title("Random Forest Accuracy for Different Max Leaf Nodes")
plt.show()

RFAcc = max(scoreListRF)
print("Random Forest Accuracy: {:.2f}%".format(RFAcc * 100))
RFcv_scores = cross_val_score(RFclassifier, X_train, y_train, cv=5)
print("Random Forest CV Scores:", RFcv_scores)

# Store model results
results = {
    'Model Name': ['Logistic Regression', 'SVC', 'Decision Tree', 'Random Forest'],
    'Mean Accuracy (%)': [LRAcc, SVCAcc, DTAcc, RFAcc]
}

# Display results as a DataFrame
df1 = pd.DataFrame(results)
print(df1)

# Load the test dataset
test_df = pd.read_csv("https://raw.githubusercontent.com/Pradeesh890/pract/refs/heads/main/exp2/2b.%20loan-test.csv")
print(test_df.head())  # Display first few rows of the test dataset
