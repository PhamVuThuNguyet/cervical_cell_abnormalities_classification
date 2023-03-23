import numpy as np
import os
from keras.applications import ResNet50V2

from keras.applications import imagenet_utils

from keras.utils import img_to_array, load_img

import random
from tqdm import tqdm

from imutils import paths

from sklearn.utils import class_weight

import joblib

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def plot_hist_acc(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show(block=True)


def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show(block=True)


# it contains path for each image in our folder
imagePaths = list(paths.list_images("../data/full_images"))
random.shuffle(imagePaths)

# it will extract the labels from the path of each image
labels = [p.split(os.path.sep)[1].split('_')[1].split('.')[0]
          for p in imagePaths]
classNames = [str(x) for x in np.unique(labels)]

# convert the labels from integers to vectors
le = LabelEncoder()
labels = le.fit_transform(labels)



# loading the EfficientNetV2L pre-trained on imagenet network
model = ResNet50V2(include_top=True, weights=None, classes=5)

print(model.summary())
print(le.classes_)

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(labels),
                                                  y=labels)


onehot_encoder = OneHotEncoder(sparse=False)
labels = labels.reshape(len(labels), 1)
labels = onehot_encoder.fit_transform(labels)

X=[]
for i in tqdm(imagePaths):
    image = load_img(i, target_size=(224, 224))  # loading image by there paths
    image = img_to_array(image)  # converting images into arrays
    # inserting a new dimension because keras need extra dimensions
    # image = np.expand_dims(image, axis=0)
    # # preprocessing image according to imagenet data
    # image = imagenet_utils.preprocess_input(image)
    X.append(image)
X = np.array(X)

model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

weights = {}
for i in range(5):
    weights[i] = class_weights[i]

hist = model.fit(x=X, y=labels, batch_size=4, epochs=200, validation_split=0.3, class_weight=weights, verbose=2)
plot_hist_acc(hist)
plot_hist_loss(hist)


if not os.path.exists('../output/models/ResNet50'):
    os.makedirs('../output/models/ResNet50')
# Save the model as a pickle in a file
# joblib.dump(model.best_estimator_, 'output/models/model_SVC_4.pkl')
joblib.dump(model, '../output/models/ResNet50/model_resnet.pkl')



