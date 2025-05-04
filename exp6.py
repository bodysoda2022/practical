import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def load_data():
    X_train_path = 'UCI HAR Dataset/train/X_train.txt'
    y_train_path = 'UCI HAR Dataset/train/y_train.txt'
    X_test_path = 'UCI HAR Dataset/test/X_test.txt'
    y_test_path = 'UCI HAR Dataset/test/y_test.txt'
    X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None)
    y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None)
    X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None)
    y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test, scaler

X_train, y_train, X_test, y_test, scaler = load_data()
num_classes = 6
y_train = to_categorical(y_train - 1, num_classes=num_classes)
y_test = to_categorical(y_test - 1, num_classes=num_classes)

def reshape_data(X):
    return X.reshape((X.shape[0], 1, X.shape[1], 1))

X_train_reshaped = reshape_data(X_train)
X_test_reshaped = reshape_data(X_test)

model = Sequential([
    Conv2D(32, (1, 5), activation='relu', input_shape=(1, X_train.shape[1], 1)),
    MaxPooling2D(pool_size=(1, 2)),
    Conv2D(64, (1, 5), activation='relu'),
    MaxPooling2D(pool_size=(1, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=64, validation_split=0.2)
model.save('har_model.h5')

activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
                   'SITTING', 'STANDING', 'LAYING']

def preprocess_image(image_path, input_shape, scaler):
    img = Image.open(image_path).convert('L')
    img = img.resize((input_shape[1], input_shape[0]))
    img_array = np.array(img).astype('float32')
    img_array /= 255.0
    img_array = img_array.flatten().reshape(1, -1)
    img_array = img_array.reshape(1, input_shape[0], input_shape[1], 1)
    return img_array

model = tf.keras.models.load_model('har_model.h5')

def predict_activity(image_path, model, input_shape, scaler):
    img_array = preprocess_image(image_path, input_shape, scaler)
    print(f'Processed image shape: {img_array.shape}')
    prediction = model.predict(img_array)
    print(f'Prediction array: {prediction}')
    activity_index = np.argmax(prediction, axis=1)[0]
    activity = activity_labels[activity_index]
    return activity

image_path = '1.jpeg'
input_shape = (1, X_train.shape[1], 1)
activity = predict_activity(image_path, model, input_shape, scaler)
print(f'Predicted activity: {activity}')
