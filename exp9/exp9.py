import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Load datasets
data_true = pd.read_csv('https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp9/9a.%20True.csv')
data_fake = pd.read_csv('https://raw.githubusercontent.com/bodysoda2022/practical/refs/heads/main/exp9/9b.%20Fake.csv')

# Adding class column (0 for fake, 1 for true)
data_fake["class"] = 0
data_true['class'] = 1

# Remove last 10 entries for manual testing
data_fake_manual_testing = data_fake.tail(10)
data_fake.drop(data_fake.tail(10).index, axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10)
data_true.drop(data_true.tail(10).index, axis=0, inplace=True)

# Reassign class labels for the manual testing part
data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

# Combine fake and true news data
data_merge = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Check for missing values
print(data.isnull().sum())

# Shuffle and reset index
data = data.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply text preprocessing
data['text'] = data['text'].apply(wordopt)

# Define features and target
x = data['text']
y = data['class']

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Text vectorization using TF-IDF
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression model
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Accuracy:", LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))

# Decision Tree model
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("Decision Tree Accuracy:", DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))

# Gradient Boosting model
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)
pred_gb = GB.predict(xv_test)
print("Gradient Boosting Accuracy:", GB.score(xv_test, y_test))
print(classification_report(y_test, pred_gb))

# Random Forest model
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)
print("Random Forest Accuracy:", RF.score(xv_test, y_test))
print(classification_report(y_test, pred_rf))

# Function to return label name from prediction
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    # Predictions
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    # Display predictions
    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction:{}".format(
        output_label(pred_LR[0]), 
        output_label(pred_DT[0]), 
        output_label(pred_GB[0]), 
        output_label(pred_RF[0])
    ))

# Take user input for manual testing
news = str(input("Enter the news text: "))
manual_testing(news)

# Test another news input
news = str(input("Enter another news text: "))
manual_testing(news)
