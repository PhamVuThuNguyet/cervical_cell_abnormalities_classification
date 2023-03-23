from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from imutils import paths
import os

from PIL import Image
import pandas as pd
import numpy as np
from imutils import paths
import os
import joblib

# Utilities for ImageNet data preprocessing & prediction decoding
from keras.applications import imagenet_utils
from keras.utils import img_to_array, load_img

import numpy as np
from PIL import Image
from keras.applications import ResNet50V2


imagePaths = list(paths.list_images("../data/full_images"))
model = joblib.load('../output/models/ResNet50/model_resnet.pkl')

classes = ['actinomyces', 'atrophy', 'candida', 'clue', 'normal']
labels = []
targets = []

for i in imagePaths:
    targets.append(i.split(os.path.sep)[1].split('_')[1].split('.')[0])

    image = load_img(i, target_size=(224, 224))  # loading image by there paths
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    data = model.predict(image)
    label = np.argmax(data)

    print(targets[len(targets) - 1], label)

    labels.append(label)

final_targets = [classes.index(i) for i in targets]

print(classification_report(final_targets, labels, target_names=classes))
