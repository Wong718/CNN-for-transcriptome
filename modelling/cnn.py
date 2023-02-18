"""
This file is part of the Verifying explainability of a deep learning tissue classifier trained on RNA-seq data project.

Verifying explainability of a deep learning tissue classifier trained on RNA-seq data project is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.


Verifying explainability of a deep learning tissue classifier trained on RNA-seq data project is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the Verifying explainability of a deep learning tissue classifier trained on RNA-seq data project.  If not, see <http://www.gnu.org/licenses/>.
"""
import keras.optimizers
import numpy as np
import math
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
# from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.optimizers import rmsprop_v2
# from sklearn.metrics import f1_score

def convert_2d(df):
    np_array = df.values
    pixel_length = math.ceil(math.sqrt(df.shape[1]))
    pad_length = pixel_length ** 2 - df.shape[1]
    np_array_padded = np.pad(np_array, ((0, 0), (0, pad_length)), mode="constant")
    np_array_2d = np.reshape(
        np_array_padded, (np_array_padded.shape[0], pixel_length, pixel_length)
    )
    np_array_2d_1 = np.expand_dims(np_array_2d, axis=3)
    return np_array_2d_1

def convert_onehot(df_labels):
    np_labels = np.array(df_labels)
    encoder = LabelBinarizer()
    np_labels_onehot = encoder.fit_transform(np_labels)
    return np_labels_onehot


def log_transform(x, label=False):
    if label == True:
        labels = x.pop("tissue")
    x = np.log2(x+0.001)
    x += 10
    if label == True:
        x["tissue"] = labels
    return x

def prepare_x_y(df, label):
    X = df.drop(label, axis=1)
    y = df.loc[:,label]
    X_log = log_transform(X)
    X_converted = convert_2d(X_log)
    y_converted = convert_onehot(y)
    return X_converted, y_converted

def run_inference(test_df, trained_model):
    X_test, _ = prepare_x_y(test_df, "tissue")
    y_test = test_df["tissue"]


    lb = LabelBinarizer()
    lb.fit(y_test.values)
    y_preds = trained_model.predict_classes(X_test)
    num_preds = len(y_preds)

    classes = test_df["tissue"].unique()
    num_classes = len(classes)

    y_preds_onehot = np.zeros([num_preds, num_classes])
    y_preds_onehot[np.arange(num_preds), y_preds] = 1

    y_preds_labels = lb.inverse_transform(y_preds_onehot)
    
    return y_preds_labels


def keras_cnn(X, y, loss_func="categorical_crossentropy"):
    model = Sequential()

    model.add(
        Conv2D(32, (3, 3), input_shape=(X.shape[1], X.shape[2], 1), padding="same")
    )
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(y.shape[1]))
    model.add(Activation("softmax"))

    model.compile(
        loss=loss_func,
        optimizer=rmsprop_v2.RMSprop(lr=0.0001, rho=0.9, decay=0.01),
        metrics=["accuracy"],
    )

    return model


