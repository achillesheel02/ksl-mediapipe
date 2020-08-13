from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
import utils
import matplotlib.pyplot as plt

from keras.layers import Flatten
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from keras.layers import ConvLSTM2D

# Initializing variables
window = utils.get_sequence_length()  # depends on time window
epochs = 200
batch_size = 16
pose_vec_dim = 42  # depends on pose estimation model used
cores = cpu_count()

class_names = utils.generate_labels().classes_
num_class = len(class_names)
lbl_dict = {class_name: idx for idx, class_name in enumerate(class_names)}


def load_data():
    n_steps, n_length = 1, window
    X, y = utils.create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, pose_vec_dim))
    X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, pose_vec_dim))
    y_train = tf.keras.utils.to_categorical(list(map(lbl_dict.get, y_train)), num_class)
    y_test = tf.keras.utils.to_categorical(list(map(lbl_dict.get, y_test)), num_class)

    return X_train, X_test, y_train, y_test


def lstm_model():
    model = Sequential()
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(window, pose_vec_dim)))
    model.add(Dropout(0.2))
    model.add((Dense(100, activation='relu')))
    model.add(Dense(len(class_names), activation='softmax'))
    print(model.summary())
    return model


def cnntd_model():
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, window, pose_vec_dim)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.7)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.7))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))
    return model


def convlstm_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(1, 1, window, pose_vec_dim)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))
    return model


if __name__ == '__main__':
    model = cnntd_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    X_train, X_test, y_train, y_test = load_data()
    lowest_loss = tf.keras.callbacks.ModelCheckpoint(filepath='model_low_loss_pitch_2.h5', mode='min', monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True)
    highest_acccuracy = tf.keras.callbacks.ModelCheckpoint(filepath='model_high_acc_pitch_2.h5', mode='max',
                                                           monitor='val_accuracy',
                                                           verbose=2,
                                                           save_best_only=True)
    callbacks_list = [lowest_loss, highest_acccuracy]

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("History keys:", (history.history.keys()))
    # summarise history for training and validation set accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # summarise history for training and validation set loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model_2_pitch_1.h5')
    print("Saved model to disk")
