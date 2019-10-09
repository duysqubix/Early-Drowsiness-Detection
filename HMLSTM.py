import sys
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import datetime


class HMLSTM:
    train_dataset = None
    test_dataset = None
    training_steps = None
    validation_steps = None
    scaler = None
    loss_history = {}

    def __init__(self, batch_size=64, time_step=30, num_features=4, d_hidden_size=64, r_hidden_size=32, activation='relu', output_activation='softmax', num_layers=4):
        self.batch_size = batch_size

        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Dense(
            d_hidden_size, input_shape=(time_step, num_features)))
        self.model.add(tf.keras.layers.LSTM(r_hidden_size, input_shape=(
            time_step, d_hidden_size), recurrent_dropout=0.4, dropout=0.4, return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(d_hidden_size, activation='relu')))

        self.model.add(tf.keras.layers.LSTM(r_hidden_size, input_shape=(
            time_step, d_hidden_size), recurrent_dropout=0.2, dropout=0.2, return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(d_hidden_size, activation='relu')))

        self.model.add(tf.keras.layers.LSTM(
            r_hidden_size, input_shape=(time_step, d_hidden_size), return_sequences=True))
        self.model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(d_hidden_size, activation='relu')))

        self.model.add(tf.keras.layers.LSTM(r_hidden_size))
        self.model.add(tf.keras.layers.Dense(
            d_hidden_size//4, activation='relu'))
        self.model.add(tf.keras.layers.Dense(
            d_hidden_size//4, activation='relu'))
        self.model.add(tf.keras.layers.Dense(
            d_hidden_size//4, activation='relu'))
        self.model.add(tf.keras.layers.Dense(
            d_hidden_size//8, activation='relu'))
        self.model.add(tf.keras.layers.Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['acc'])
        self.model.summary()

    def scale_data(self, data, scaler_dict=None, copy=True):
        if copy:
            data = data.copy()

        if scaler_dict is None:
            # assuming passing training data, fit supplied training data and return both scaled data AND scalers
            scaler_dict = {}
            for feature in range(data.shape[-1]):
                scaler_dict["scaler_feature:{}".format(
                    feature)] = MinMaxScaler(feature_range=(0, 1))
                data[:, :, feature] = scaler_dict["scaler_feature:{}".format(
                    feature)].fit_transform(data[:, :, feature])
            return data, scaler_dict
        else:
            # assuming all other data is passed, simply scale with supplied scaler_dict and return data
            assert isinstance(scaler_dict, dict)

            for feature in range(data.shape[-1]):
                data[:, :, feature] = scaler_dict["scaler_feature:{}".format(
                    feature)].transform(data[:, :, feature])
            return data, scaler_dict

    def _parse_labels(self, label):
        label[label == 10] = 2
        label[label == 5] = 1
        label[label == 0] = 0
        label = tf.keras.utils.to_categorical(label)
        return label

    def compile_dataset(self, blinks_path, labels_path, scale=True):
        blinks_train = np.load("{}.npy".format(
            blinks_path[0])).astype(np.float32)
        blinks_test = np.load("{}.npy".format(
            blinks_path[1])).astype(np.float32)

        labels_train = self._parse_labels(
            np.load("{}.npy".format(labels_path[0])).astype(np.uint8))

        labels_test = self._parse_labels(
            np.load("{}.npy".format(labels_path[1])).astype(np.uint8))

        if scale:
            blinks_train, scaler_dict = self.scale_data(
                blinks_train, scaler_dict=None)
            blinks_test, scaler_dict = self.scale_data(
                blinks_test, scaler_dict=scaler_dict)

        self.training_steps = blinks_train.shape[0]//self.batch_size
        self.validation_steps = blinks_test.shape[0]//self.batch_size

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (blinks_train, labels_train)).batch(self.batch_size).repeat()

        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (blinks_test, labels_test)).batch(self.batch_size).repeat()

        return blinks_train, labels_train

    def fit(self, epochs=5, **kwargs):
        history = self.model.fit(self.train_dataset, epochs=epochs, steps_per_epoch=self.training_steps,
                                 validation_steps=self.validation_steps, shuffle=True, validation_data=self.test_dataset)
        self.loss_history["fold_{}".format(kwargs['fold'])] = history.history

    def kfold_fit(self, num_folds=5, epochs=80):
        self.loss_history = {}
        for fold in range(num_folds):
            fold += 1  # to line up with recorded blink events

            # get data to train
            blink_train_path = "folds/Blinks_30_Fold{}".format(fold)
            blink_test_path = "folds/BlinksTest_30_Fold{}".format(fold)

            blink_train_label = "folds/Labels_30_Fold{}".format(fold)
            blink_test_label = "folds/LabelsTest_30_Fold{}".format(fold)

            blinks_path, labels_path = (
                blink_train_path, blink_test_path), (blink_train_label, blink_test_label)
            print(blinks_path)
            print(labels_path)
            # update internal datasets:
            self.compile_dataset(blinks_path=blinks_path,
                                 labels_path=labels_path, scale=True)

            # now train:
            self.fit(epochs=epochs, fold=fold)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # save loss_history
        with open("model_and_weights/{}_loss.history".format(current_time), "wb") as f:
            pickle.dump(self.loss_history, f)
            print("Saved Loss History")

        # save weights of the model
        file_name_weights = "{}_weights.h5".format(current_time)
        self.model.save_weights(
            "model_and_weights/{}".format(file_name_weights))

        # save model json as pickle
        file_name_model = "{}_model.json".format(current_time)
        with open("model_and_weights/{}".format(file_name_model), "wb") as f:
            pickle.dump(self.model.to_json(), f)


def compare(pred, actual, num=4):
    classes = [0, 5, 10]
    for i in range(num):
        i = np.random.randint(0, len(actual))
        max_ = np.argmax(pred[i])
        max_true = np.argmax(actual[i])
        print("Predicted: {}\t Actual: {}".format(
            classes[max_], classes[max_true]))


if __name__ == '__main__':
    model = HMLSTM(output_activation='softmax', activation='relu')
    print(model.model.optimizer)
    model.kfold_fit(epochs=80)
