import os
import random
from itertools import product
from collections import Counter

# Data processing
import numpy as np
import pandas as pd

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Image processing
import cv2
from imutils import contours


# Set seeds for reproducibility
SEED = 60
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# TASK 1

# Load Data
with np.load('./data/training-dataset.npz') as data:
    img = data['x']
    lbl = data['y']


# Background
def convert_label(label):
    '''Returns the letter that is associated with the label'''
    return chr(label+64)


def plot_example_images(img):
    '''Plots a grid of random examples images'''
    ROWS, COLS = 3, 10

    fig, axes = plt.subplots(ROWS, COLS,
                             figsize=(COLS, ROWS),
                             facecolor='w')

    for i in range(ROWS):
        for j in range(COLS):
            index = np.random.randint(len(img))
            image = img[index].reshape(28, 28)
            label = convert_label(lbl[index])

            axes[i, j].imshow(image, cmap='binary')
            axes[i, j].text(0, 0, label)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('./figures/examples.png', dpi=300)


plot_example_images(img)


# Evaluation functions


def save_model(model, model_name):
    '''Saves model weights and architecture'''
    model_json = model.to_json()
    with open(f'./models/{model_name}.json', 'w') as f:
        f.write(model_json)
    model.save_weights(f'./models/{model_name}.h5')
    print(f'Model saved as {model_name}.')


def load_model(model_name):
    with open(f'./models/{model_name}.json') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)
    model.load_weights(f'./models/{model_name}.h5')
    print(f'Loaded model {model_name}.')
    return model


def plot_model_history(history, model_name):
    '''Plots the model history by epoch'''
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='w')
    (pd.DataFrame(history.history)
        [['accuracy', 'val_accuracy']]
        .plot(ax=ax,
              style=[':', '-'],
              color=['#0066bf', '#0066bf'],
              marker='.'))

    # Titles and labels
    plt.xlabel(r'Epoch', fontsize=16)
    ax.legend(['Training Accuracy', 'Validation Accuracy'],
              loc=5, fontsize=14)

    plt.grid(axis='x')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Limits
    num_epochs = len(history.history['accuracy'])
    plt.xlim(0, num_epochs-1)
    plt.ylim(0.6, 1)

    # Set ticks
    plt.xticks(range(num_epochs), labels=range(1, num_epochs+1), fontsize=13)
    y_ticks = np.arange(0.6, 1.0, 0.05)
    plt.yticks(y_ticks, labels=[f'{l:.0%}' for l in y_ticks], fontsize=13)

    plt.savefig(f'./figures/{model_name}_history.png',
                dpi=300, bbox_inches='tight')


def plot_confusion_matrix(y_test_lbl, y_pred_lbl, model_name):
    '''Plots confusion matrix with model accuracy by letter on diagonal'''
    # Create confusion matrix
    cm = np.zeros((26, 26))
    for t, p in zip(y_test_lbl, y_pred_lbl):
        cm[t, p] += 1
    cm /= cm.sum(axis=1)  # Convert to percentages

    # Custom annotation
    annot = cm/cm.sum(axis=1)
    annot = annot.round(2)
    annot[annot < 0.1] = 0.00
    annot = annot.astype('str')
    annot[annot == '0.0'] = ''

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='w')
    g = sns.heatmap(cm, ax=ax, cmap='Blues', center=0.75, cbar=False,
                    linecolor='#ededed', linewidths=.005,
                    annot=annot, fmt='', annot_kws={'fontsize': 12})

    # X ticks to top
    g.xaxis.set_ticks_position('top')
    g.tick_params(axis='both', which='both', length=0)

    # Change labels to letters
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    g.set_xticklabels(letters)
    g.set_yticklabels(letters)

    plt.savefig(f'./figures/{model_name}_confusion_matrix.png',
                dpi=300, bbox_inches='tight')


def plot_misclassified_images(model, model_name):
    '''Plots grid of random misclassified images'''
    ROWS, COLS = 2, 10

    fig, axes = plt.subplots(ROWS, COLS,
                             figsize=(COLS, ROWS),
                             facecolor='w')

    for i in range(ROWS):
        for j in range(COLS):
            index = np.random.randint(len(X_test_wrong))
            image = X_test_wrong[index].reshape(28, 28)

            prediction = np.argmax(model.predict(image.reshape(1, 28, 28, 1)))
            actual_label = convert_label(y_test_wrong[index]+1)
            predicted_label = convert_label(prediction+1)

            axes[i, j].imshow(image, cmap='binary')
            axes[i, j].text(0, 0, actual_label, c='k')
            axes[i, j].text(7, 0, predicted_label, c='#0066bf')
            axes[i, j].axis('off')

    plt.savefig(f'./figures/{model_name}_wrong_predictions.png',
                dpi=300, bbox_inches='tight')


'''
PREPROCESS DATA
- Normalize and reshape images.
- Onehot encode labels.
'''

# Define constants
N_IMAGES = len(img)
IMAGE_SIZE = int(np.sqrt(img[0].size))
N_CLASSES = len(np.unique(lbl))

EPOCHS = 20


def preprocess_images(img, num_images, image_size):
    '''Returns normalized and reshaped image'''
    img = tf.keras.utils.normalize(img, axis=1)
    img = img.reshape(num_images, image_size[0], image_size[1], 1)
    return img


def preprocess_labels(lbl, num_classes):
    '''Returns encoded labels'''
    lbl = tf.keras.utils.to_categorical(lbl-1, num_classes)
    return lbl


X = preprocess_images(img, N_IMAGES, (IMAGE_SIZE, IMAGE_SIZE))
y = preprocess_labels(lbl, N_CLASSES)
print('X shape:', X.shape)
print('y shape:', y.shape)

# Set callbacks
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3,
                              factor=0.1, min_lr=0.00001)


'''
CONVOLUTIONAL NEURAL NETWORK
'''

# Split train and test data
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.15, random_state=SEED)


# Split train data into training and validations sets
BATCH_SIZE = 64
data_generator = \
    tf.keras.preprocessing.image.ImageDataGenerator(validation_split=.15)

training_data_generator = data_generator.flow(X_train,
                                              y_train,
                                              subset='training',
                                              batch_size=BATCH_SIZE,
                                              seed=SEED)

validation_data_generator = data_generator.flow(X_train,
                                                y_train,
                                                subset='validation',
                                                batch_size=BATCH_SIZE,
                                                seed=SEED)


def build_cnn_model():
    kernel_initializer = \
        tf.keras.initializers.glorot_uniform(seed=SEED)

    model = tf.keras.Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        Conv2D(64, 5, strides=2, padding='same',
               kernel_initializer=kernel_initializer),
        LeakyReLU(),
        Conv2D(32, 2, strides=1, padding='same',
               kernel_initializer=kernel_initializer),
        LeakyReLU(),
        MaxPooling2D(pool_size=2, strides=1),
        Dropout(0.5, seed=SEED),

        Conv2D(64, 5, strides=2, padding='same',
               kernel_initializer=kernel_initializer),
        LeakyReLU(),
        Conv2D(32, 2, strides=1, padding='same',
               kernel_initializer=kernel_initializer),
        LeakyReLU(),
        MaxPooling2D(pool_size=2, strides=1),
        Dropout(0.5, seed=SEED),

        Flatten(),
        Dense(128, kernel_initializer=kernel_initializer),
        LeakyReLU(),
        Dropout(0.3, seed=SEED),
        Dense(128, kernel_initializer=kernel_initializer),
        LeakyReLU(),
        Dense(N_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


cnn = build_cnn_model()
print(cnn.summary())


# Train model
def train_cnn(model, epochs, callbacks):
    history = model.fit(training_data_generator,
                        epochs=epochs,
                        validation_data=validation_data_generator,
                        callbacks=callbacks)

    _, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Model accuracy:', test_acc)
    return history


EPOCHS = 20
history = train_cnn(cnn, EPOCHS, callbacks=[early_stopping, reduce_lr])
save_model(cnn, model_name='cnn')


# Plot training and validation accuracy by epoch
plot_model_history(history, model_name='cnn')


# Confusion matrix
y_test_lbl = np.argmax(y_test, axis=1)
y_pred_lbl = np.argmax(cnn.predict(X_test), axis=1)
plot_confusion_matrix(y_test_lbl, y_pred_lbl, model_name='cnn')


# Evaluate wrong predictions
wrong_predictions = y_test_lbl != y_pred_lbl
X_test_wrong = X_test[wrong_predictions]
y_test_wrong = y_test_lbl[wrong_predictions]
print(pd.Series(
    [convert_label(y+1) for y in y_test_wrong])
    .value_counts().to_frame('Misclassified').T)


# Grid of random wrong predictions
plot_misclassified_images(cnn, model_name='cnn')


'''
CORRUPTED IMAGE PREDICTION
1. Denoise image
2. Seperate images
3. Clean images
4. Predict images
'''


class PredictImage:
    def __init__(self, image, model):
        self.raw_image = image
        self.model = model
        self.denoised_image = self._denoise_image()
        self.cnts = self._find_contours()
        self.cnts_images = self._contours_to_images()
        self.letter_images = self._clean_images()
        self.letter_probabilities = self.get_letter_probabilities()
        self.top_predictions = self.get_5_predictions()

    def _denoise_image(self):
        '''Returns the the denoised image'''
        denoised_image = self.raw_image.copy()
        denoised_image = cv2.convertScaleAbs(denoised_image)
        mask = np.zeros(denoised_image.shape, dtype=np.uint8)
        blur = cv2.GaussianBlur(denoised_image, (7, 7), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised_image[thresh == 0] = 0
        return denoised_image

    def _find_contours(self, min_area=25):
        '''Returns a contours object with the contrours'''
        cnts = cv2.findContours(self.denoised_image,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = [c for c in cnts if min_area < cv2.contourArea(c)]
        if len(cnts) > 4:
            cnts = sorted(cnts, key=cv2.contourArea)[-4:]
        cnts, _ = contours.sort_contours(cnts, method='left-to-right')
        return cnts

    def _contours_to_images(self):
        '''Converts the letter contours to images'''
        contour_images = []
        x_prev = 0
        for c in self.cnts:
            x, _, w, _ = cv2.boundingRect(c)
            x_start, x_end = x, x+w
            image = self.denoised_image[1:29, x_start:x_end]
            if x_prev == 0 or x_start > x_prev:
                contour_images.append(image)
            else:
                prev_image = contour_images[-1]
                contour_images[-1] = np.hstack([prev_image,
                                                self.denoised_image[1:29, x_prev:x_end]])
            x_prev = x_end
        return contour_images

    def _reshape_image(self, image):
        '''Returns padded image with shape 28x28'''
        width = image.shape[1]
        if width >= 28:
            return image[:, 0:28]
        else:
            pad = 28 - width
            pad_left = np.ceil(pad/2).astype('int')
            pad_right = np.floor(pad/2).astype('int')
            return np.pad(image, ((0, 0), (pad_left, pad_right)))

    def _split_image(self, image, n):
        '''
        Split image evenly in n images. Attemps to split the image at a gap
        between the letters. Othewise splits image naively in equal parts.
        '''
        split_point = np.argmin(image.sum(axis=0))
        if (n == 2 and image[:, split_point].sum() < 250 and
                0.2 < (split_point/image.shape[1]) < 0.8):
            return image[:, :split_point], image[:, split_point:]
        else:
            return np.array_split(image, n, axis=1)

    def _split_images(self, images):
        '''Splits the wide images in multiple images to return 4 images'''
        num_images = len(images)

        image_width = [i.shape[1] for i in images]
        images_to_split = len([w for w in image_width if w > 28])
        if images_to_split == 0:
            images_to_split = 1

        splits = (4-num_images) / images_to_split
        for _ in range(images_to_split):
            image_width = [i.shape[1] for i in images]
            large_image_index = np.argsort(image_width)[-1]
            image = images[large_image_index]
            images[large_image_index:large_image_index +
                   1] = self._split_image(image, splits+1)
        return images

    def _clean_images(self):
        '''Returns 4 images of shape 28x28.'''
        images = self.cnts_images.copy()
        if len(images) < 4:
            images = self._split_images(images)

        letter_images = []
        for image in images:
            image = self._reshape_image(image)
            image = tf.keras.utils.normalize(image, axis=1)
            letter_images.append(image)
        return letter_images

    def get_letter_probabilities(self):
        '''
        Returns dictionary with the 5 most probable letters predicted by the
        model for the four images.
        '''
        letter_probabilities = []
        for i, image in enumerate(self.letter_images):
            probabilities = self.model.predict(image.reshape(1, 28, 28, 1))[0]
            sorted_probabilities = np.sort(probabilities)[::-1]
            sorted_index = np.argsort(probabilities)[::-1]
            predictions = {k+1: v for k,
                           v in zip(sorted_index, sorted_probabilities)}
            letter_probabilities.append(
                dict(Counter(predictions).most_common(5)))
        return letter_probabilities

    def get_5_predictions(self):
        '''
        Returns the 5 most probable four letter combinations predicted by the
        model'''
        combinations = list(product(*[d.values()
                                      for d in self.letter_probabilities]))
        letter_index = list(product(*self.letter_probabilities))

        top_5_predictions = []
        for i in range(5):
            probs = np.argsort(np.prod(combinations, axis=1))[::-1][i]
            letters = letter_index[probs]
            pred = ''.join(str(l).zfill(2) for l in letters)
            top_5_predictions.append(pred)
        return top_5_predictions

    def convert_to_letters(self):
        '''Returns a string of four letters that corresponds to the predicted label'''
        letters = []
        for l in self.top_predictions:
            ints = [int(l[i:i+2]) for i in range(0, len(l), 2)]
            lbls = [chr(i+64) for i in ints]
            letters.append(''.join(lbls))
        return letters

    def plot_raw_image(self):
        '''Plots original image'''
        plt.imshow(self.raw_image, cmap='binary')
        plt.axis('off')
        plt.show()

    def plot_denoised_image(self):
        '''Plots denoised image'''
        plt.imshow(self.denoised_image, cmap='binary')
        plt.axis('off')
        plt.show()

    def plot_cnts_images(self):
        '''Plots the images of the letter contours'''
        fig, axes = plt.subplots(ncols=len(self.cnts_images), facecolor='k')
        for i, img in enumerate(self.cnts_images):
            axes[i].imshow(img, cmap='binary')
            axes[i].set_title(str(i+1), c='w', fontsize=16)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def plot_letter_images(self):
        '''Plots the processed letter images'''
        fig, axes = plt.subplots(ncols=len(self.letter_images), facecolor='k')
        for i, img in enumerate(self.letter_images):
            axes[i].imshow(img, cmap='binary')
            axes[i].set_title(str(i+1), c='w', fontsize=16)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def plot_prediction_image(self):
        '''Plots the raw image with the predictions as title'''
        plt.imshow(self.raw_image, cmap='binary')
        plt.title(' | '.join(self.convert_to_letters()))
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# Load data
img_test = np.load('./data/test-dataset.npy')


# Make predictions
def predict_images(images, model):
    predictions = pd.DataFrame()
    for i, image in enumerate(images):
        pi = PredictImage(image, model)
        prediction = pd.DataFrame([pi.top_predictions], index=[i])
        predictions = pd.concat([predictions, prediction])

        if i % 500 == 0:
            print(f'[{i}/{len(images)}]\tPredicting images...')
    print('Done.')
    return predictions


# Make predictions
predictions = predict_images(img_test, cnn)
print(predictions.head())
