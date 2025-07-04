import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Set up directories
os.makedirs('model', exist_ok=True)
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Image and training parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data Augmentation & Preprocessing
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=15,
                               zoom_range=0.1,
                               horizontal_flip=True
                              ).flow_from_directory(train_dir,
                                                   target_size=IMAGE_SIZE,
                                                   class_mode='categorical')
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir,
                                                                 target_size=IMAGE_SIZE,
                                                                 class_mode='categorical')

# Build a VGG16-based model
base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base.layers:
    layer.trainable = False

x = Flatten()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save the best model during training
checkpoint = ModelCheckpoint('model/cleantech_vgg.h5',
                             save_best_only=True,
                             monitor='val_accuracy',
                             mode='max')

# Train
model.fit(train_gen,
          validation_data=val_gen,
          epochs=EPOCHS,
          callbacks=[checkpoint])

print("✅ Training complete! Model saved to 'model/cleantech_vgg.h5'")
