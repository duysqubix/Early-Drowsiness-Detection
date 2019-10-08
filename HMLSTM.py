import sys
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class HMLSTM:
    train_dataset = None
    test_dataset = None
    training_steps = None
    validation_steps = None
    scaler = None

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

    def compile_dataset(self, scale=True):
        blinks_train = np.load("data_train.npy").astype(np.float32)
        blinks_test = np.load("data_test.npy").astype(np.float32)

        labels_train = self._parse_labels(
            np.load("labels_train.npy").astype(np.uint8))

        labels_test = self._parse_labels(
            np.load("labels_test.npy").astype(np.uint8))

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

    def fit(self, epochs=5):
        self.model.fit(self.train_dataset, epochs=epochs, steps_per_epoch=self.training_steps,
                       validation_steps=self.validation_steps, shuffle=True, validation_data=self.test_dataset)


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
    model.model.summary()
    print(model.model.optimizer)
    x, y = model.compile_dataset(scale=True)
    if '-w' in sys.argv:
        try:
            model.model.load_weights("model.weights")
            print("loaded weights")
        except FileNotFoundError:
            pass
    model.fit(epochs=5)
    model.model.save_weights("model.weights")

    pred = model.model.predict(x)
    compare(pred, y)
