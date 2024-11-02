# Importing required libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize parameters for the model
initial_learning_rate = 1e-4
epochs = 20
batch_size = 32

# Set the directory containing the dataset
dataset_directory = r"C:\Users\likki\Mask Detection\CODE\Face-Mask-Detection-master\dataset"
categories = ["with_mask", "without_mask"]

# Loading images from the dataset
print("[INFO] Loading images...")

image_data = []
image_labels = []

for category in categories:
    category_path = os.path.join(dataset_directory, category)
    for image_filename in os.listdir(category_path):
        image_path = os.path.join(category_path, image_filename)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        image_data.append(image)
        image_labels.append(category)

# One-hot encode the labels
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(image_labels)
image_labels = to_categorical(image_labels)

# Convert lists to NumPy arrays
image_data = np.array(image_data, dtype="float32")
image_labels = np.array(image_labels)

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(image_data, image_labels,
                                                  test_size=0.20, stratify=image_labels, random_state=42)

# Configure the image data generator for augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load the MobileNetV2 model without the top layer
base_model = MobileNetV2(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

# Define the custom head model for classification
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# Combine the base model and head model
final_model = Model(inputs=base_model.input, outputs=head_model)

# Freeze the base model layers for initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
print("[INFO] Compiling model...")
optimizer = Adam(learning_rate=initial_learning_rate, decay=initial_learning_rate / epochs)
final_model.compile(loss="binary_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])

# Train the model
print("[INFO] Training model...")
history = final_model.fit(
    data_augmentation.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=(testX, testY),
    validation_steps=len(testX) // batch_size,
    epochs=epochs)

# Evaluate the model
print("[INFO] Evaluating model...")
predictions = final_model.predict(testX, batch_size=batch_size)

# Determine the predicted class for each test image
predicted_classes = np.argmax(predictions, axis=1)

# Generate a classification report
print(classification_report(testY.argmax(axis=1), predicted_classes,
                            target_names=label_binarizer.classes_))

# Save the trained model to disk
print("[INFO] Saving model...")
final_model.save("mask_detector.model", save_format="h5")

# Plotting training history
num_epochs = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
