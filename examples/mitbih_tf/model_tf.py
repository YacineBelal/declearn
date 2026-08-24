import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="Models", name="tinyCNN")
class tinyCNN(keras.Model):
    def __init__(self, matched_filters, trainable_conv=True, **kwargs):
        super().__init__(**kwargs)
        self.trainable_conv = trainable_conv
        self.matched_filters = tf.Variable(
            matched_filters, dtype=tf.float32, trainable=trainable_conv
        )
        self.n_filters = matched_filters.shape[0]
        self.window_len = matched_filters.shape[2]

        self.conv = keras.models.Sequential(
            [
                keras.layers.Conv1D(
                    filters=self.n_filters,
                    kernel_size=self.window_len,
                    padding="same",
                    name="conv1d",
                ),
                keras.layers.BatchNormalization(name="batch_normalization"),
                keras.layers.Activation("tanh"),
                keras.layers.GlobalMaxPooling1D(data_format="channels_last"),
            ],
            name="conv_path",
        )

        self.rr_path = keras.models.Sequential(
            [
                keras.layers.Dense(32, activation="relu", name="dense_rr1"),
                keras.layers.Dense(16, activation="relu", name="dense_rr2"),
                keras.layers.Dense(8, activation="relu", name="dense_rr3"),
            ],
            name="rr_path",
        )

        self.merger = keras.layers.Dense(3, activation="softmax", name="merger_dense") 
        
    def call(self, inputs):
        x = inputs[0]
        rr = inputs[1]

        conv_out = self.conv(x)
        rr_path_out = self.rr_path(rr)

        return self.merger(keras.layers.concatenate([conv_out, rr_path_out]))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "matched_filters": keras.saving.serialize_keras_object(
                    self.matched_filters
                ),
                "trainable_conv": self.trainable_conv,
            }
        )
        return {**config}


    def build(self, input_shape):
        ecg_shape, rr_shape = input_shape 
        dummy_ecg = tf.zeros((1,) + tuple(ecg_shape[1:]))
        dummy_rr = tf.zeros((1,) + tuple(rr_shape[1:]))
        self.call([dummy_ecg, dummy_rr])
        super().build(input_shape)

    @classmethod
    def from_config(cls, config):
        config["matched_filters"] = keras.saving.deserialize_keras_object(config["matched_filters"])
        return cls(**config)
