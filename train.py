from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.utils import to_categorical, plot_model # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense # type: ignore
from tensorflow.keras import backend as K # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob


epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

image_files = [f for f in glob.glob(r'dataset' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

total_files = len(image_files)
print(f"Total image files found: {total_files}")

## Convert images to arrays and label categories
# for i, img in enumerate(image_files, 1):
#     print(f"Processing file {i}/{total_files}: {img}")

#     image = cv2.imread(img)
#     image = cv2.resize(image, (img_dims[0], img_dims[1]))
#     image = img_to_array(image)
#     data.append(image)

#     # Extract label from directory name
#     label = img.split(os.path.sep)[-2]
#     label = 1 if label == "woman" else 0
#     labels.append([label])


# output_dir = "preprocessed_data"
# os.makedirs(output_dir, exist_ok=True)
# np.save(os.path.join(output_dir, "data.npy"), np.array(data), allow_pickle=False)
# np.save(os.path.join(output_dir, "labels.npy"), np.array(labels), allow_pickle=False)

# print("Data and labels successfully saved.")


data = np.load("preprocessed_data/data.npy")/255.0
labels = np.load("preprocessed_data/labels.npy")


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model


model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)


model.save('gender_detection.h5')


# plt.style.use("ggplot")
# plt.figure()
# N = epochs
# plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
# # plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper right")

# # save plot to disk
# plt.savefig('plot.png')


