import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from tqdm import tqdm

sm.set_framework('tf.keras')

def data_loader(folder_dir):
    image_dataset = []
    for image_name in os.listdir(folder_dir):
        image = cv2.imread(os.path.join(folder_dir, image_name), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = np.array(image)
        image_dataset.append(image)
    return np.array(image_dataset)

image_dataset_path = input("Enter the path to the image dataset: ")
mask_dataset_path = input("Enter the path to the mask dataset: ")
mask_labels_path = input("Enter the path to the mask labels CSV file: ")

image_dataset = data_loader(image_dataset_path)
mask_dataset = data_loader(mask_dataset_path)

image_number = random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(mask_dataset[image_number])
plt.show()

mask_labels = pd.read_csv(mask_labels_path)
print(mask_labels)

def rgb_to_labels(img, mask_labels):
    label_seg = np.zeros(img.shape[:2], dtype=np.uint8)
    for i in range(mask_labels.shape[0]):
        label_seg[np.all(img == list(mask_labels.iloc[i, 1:4]), axis=-1)] = i
    return label_seg

labels = [rgb_to_labels(mask, mask_labels) for mask in mask_dataset]
labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))

image_number = random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:, :, 0])
plt.show()

n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_prepr, y_train, batch_size=16, epochs=20, verbose=1,
                    validation_data=(X_test_prepr, y_test))

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'y', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('resnet_backbone.hdf5')
model = tf.keras.models.load_model('resnet_backbone.hdf5', compile=False)

y_pred = model.predict(X_test_prepr)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = model.predict(test_img_input)
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(16, 12))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()
