from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

base = VGG16(include_top=False, weights=None, input_shape=(224,224,3))
x = Flatten()(base.output)
x = Dense(3, activation='softmax')(x)
model = Model(inputs=base.input, outputs=x)
model.save("vgg16.h5")
print("Dummy model created.")
