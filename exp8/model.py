import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from flask import Flask, render_template, request

# Load the dataset
df = pd.read_csv('students_placement.csv')

# Data processing
X = df.drop(columns=['placed'])
y = df['placed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# StandardScaler for feature scaling
scaler = StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)

# Train the models
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_trf, y_train)
log_reg_accuracy = accuracy_score(y_test, log_reg_model.predict(X_test_trf))

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train)
svc_accuracy = accuracy_score(y_test, svc_model.predict(X_test))

# Pickle the best model, here we are using SVC as the chosen model
pickle.dump(svc_model, open('model.pkl', 'wb'))

# Flask setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    cgpa = float(request.form.get('cgpa'))
    iq = int(request.form.get('iq'))
    profile_score = int(request.form.get('profile_score'))
    
    # Prepare data for prediction
    input_data = np.array([cgpa, iq, profile_score]).reshape(1, 3)
    
    # Make prediction using the pickled model
    model = pickle.load(open('model.pkl', 'rb'))
    result = model.predict(input_data)
    
    # Check prediction result
    if result[0] == 1:
        result = 'Placed'
    else:
        result = 'Not Placed'
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
