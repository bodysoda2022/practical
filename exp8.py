import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def preprocess_bw_image(image_path, size=(256, 256)):
    image = Image.open(image_path).convert('L')
    image = image.resize(size)
    image = np.array(image) / 127.5 - 1
    image = np.expand_dims(image, axis=-1)
    return image

def preprocess_color_image(image_path, size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(size)
    image = np.array(image) / 127.5 - 1
    return image

def load_image_pairs(bw_dir, color_dir, size=(256, 256)):
    bw_images = []
    color_images = []
    for filename in os.listdir(bw_dir):
        bw_path = os.path.join(bw_dir, filename)
        color_path = os.path.join(color_dir, filename)
        if os.path.exists(color_path):
            bw_images.append(preprocess_bw_image(bw_path, size))
            color_images.append(preprocess_color_image(color_path, size))
    return np.array(bw_images), np.array(color_images)

train_bw_dir = 'dataset/train/bw'
train_color_dir = 'dataset/train/color'
test_bw_dir = 'dataset/test/bw'
test_color_dir = 'dataset/test/color'

train_bw_images, train_color_images = load_image_pairs(train_bw_dir, train_color_dir)
test_bw_images, test_color_images = load_image_pairs(test_bw_dir, test_color_dir)

print(f'Train BW Images: {train_bw_images.shape}')
print(f'Train Color Images: {train_color_images.shape}')
print(f'Test BW Images: {test_bw_images.shape}')
print(f'Test Color Images: {test_color_images.shape}')

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
generator.compile(optimizer='adam', loss='mean_squared_error')
generator.fit(train_bw_images, train_color_images, epochs=50, batch_size=1)
generator.evaluate(test_bw_images, test_color_images)

def generate_and_save_images(model, test_input, save_dir):
    predictions = model.predict(test_input)
    predictions = (predictions + 1) / 2.0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(predictions)):
        img = predictions[i]
        img = tf.image.convert_image_dtype(img, tf.uint8)
        img = Image.fromarray(img.numpy())
        img.save(os.path.join(save_dir, f'colorized_image_{i}.png'))

output_dir = 'output/test_colorized'
# Generate and save the images
generate_and_save_images(generator, test_bw_images, output_dir)

# Generate and display images for comparison
def display_image_comparison(model, test_bw_images, test_color_images, num_images=3):
    predictions = model.predict(test_bw_images)
    predictions = (predictions + 1) / 2.0  # De-normalize to [0, 1]
    test_color_images = (test_color_images + 1) / 2.0
    plt.figure(figsize=(15, 5 * num_images))
    for i in range(num_images):
        plt.subplot(num_images, 3, 3*i + 1)
        plt.imshow(test_bw_images[i].reshape(256, 256), cmap='gray')
        plt.title('Black and White')
        plt.axis('off')
        plt.subplot(num_images, 3, 3*i + 2)
        plt.imshow(test_color_images[i])
        plt.title('Actual Color')
        plt.axis('off')
        plt.subplot(num_images, 3, 3*i + 3)
        plt.imshow(predictions[i])
        plt.title('Generated Color')
        plt.axis('off')
    plt.show()

def calculate_accuracy(model, test_bw_images, test_color_images):
    predictions = model.predict(test_bw_images)
    predictions = (predictions + 1) / 2.0
    test_color_images = (test_color_images + 1) / 2.0
    mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(test_color_images, predictions))
    accuracy = 1 - mse.numpy()  # Accuracy = 1 - MSE (simplified)
    return accuracy

def display_confusion_matrix(model, test_bw_images, test_color_images):
    predictions = model.predict(test_bw_images)
    predictions = (predictions + 1) / 2.0
    test_color_images = (test_color_images + 1) / 2.0
    y_true = test_color_images.flatten()
    y_pred = predictions.flatten()
    cm = confusion_matrix(y_true > 0.5, y_pred > 0.5)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Black', 'White'], yticklabels=['Black', 'White'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

accuracy = calculate_accuracy(generator, test_bw_images, test_color_images)
print(f'Model Accuracy: {accuracy:.4f}')
display_image_comparison(generator, test_bw_images, test_color_images, num_images=3)
display_confusion_matrix(generator, test_bw_images, test_color_images)
