import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os

# -----------------------
# Dataset Paths
# -----------------------
train_dir = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset\train"
valid_dir = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset\valid"
test_dir  = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\Dataset\test"

# -----------------------
# Image Generators
# -----------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

valid_gen = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# -----------------------
# CNN Model
# -----------------------
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Gender: Male/Female
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------
# Train Model
# -----------------------
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=15
)

# -----------------------
# Evaluate
# -----------------------
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -----------------------
# Save Model
# -----------------------
MODEL_PATH = r"C:\Users\Afnan khan\Desktop\Handwriting-based gender classification\project\cnn_gender_model.h5"
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
