import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    r'train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    r'test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('my_model.h5')

class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}
print("Class Labels:")
for index, label in class_labels.items():
    print(f"Class {index}: {label}")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = load_model('my_model.h5')
class_labels = {0: 'cat', 1: 'dog'}

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    return predictions

img_path = "cat_image.jpg"
predictions = predict_image(img_path)

if model.output_shape[1] == 1:
    predicted_class_index = 1 if predictions[0] > 0.5 else 0
else:
    predicted_class_index = np.argmax(predictions[0])

predicted_class_label = class_labels.get(predicted_class_index, 'Unknown Class')
print(f"Prediction: {predicted_class_label}")

def display_image_with_box(img_path, bbox=None):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

bbox_normalized = (0.8, 0.8, 0.5, 0.5)
original_img = cv2.imread(img_path)
img_height, img_width, _ = original_img.shape
x_center, y_center, width, height = bbox_normalized
x_center *= img_width
y_center *= img_height
width *= img_width
height *= img_height
x = int(x_center - width/0.9)
y = int(y_center - height/0.8)
bbox = (x, y, int(width), int(height))
display_image_with_box(img_path, bbox)