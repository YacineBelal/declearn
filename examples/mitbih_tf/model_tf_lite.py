import keras
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras

n_filters = 64


@keras.saving.register_keras_serializable()
class tinyCNN(tf_keras.Model):
    def __init__(
        self, matched_filters: np.ndarray, trainable_conv=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.trainable_conv = trainable_conv
        self.matched_filters = tf.Variable(
            matched_filters, dtype=tf.float32, trainable=trainable_conv
        )
        self.n_filters = matched_filters.shape[0]
        self.window_len = matched_filters.shape[2]

        self.conv = tf_keras.models.Sequential(
            [
                tf_keras.layers.Conv1D(
                    filters=self.n_filters,
                    kernel_size=self.window_len,
                    padding="same",
                ),
                tf_keras.layers.BatchNormalization(),
                tf_keras.layers.Activation("tanh"),
                tf_keras.layers.GlobalMaxPooling1D(
                    data_format="channels_last"
                ),
            ]
        )

        self.rr_path = tf_keras.models.Sequential(
            [
                tf_keras.layers.Dense(32, activation="relu"),
                tf_keras.layers.Dense(16, activation="relu"),
                tf_keras.layers.Dense(8, activation="relu"),
            ]
        )

        self.merger = tf_keras.Sequential(
            [
                tf_keras.layers.Concatenate(),
                tf_keras.layers.Dense(3, activation="softmax"),
            ]
        )

    # @tf.function(
    #         input_signature=[
    #             [
    #                 tf.TensorSpec(shape=[None, 1, 64], dtype=tf.float32),
    #                 tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
    #             ],
    #         ],
    #     )
    def call(self, inputs, training=False):
        x = inputs[0]
        rr = inputs[1]
        conv_out = self.conv(x)
        rr_path_out = self.rr_path(rr)

        return self.merger([conv_out, rr_path_out])

    def compile(self, optimizer, loss, **kwargs):
        super().compile(optimizer, loss, **kwargs)
        self.loss_fn = tf_keras.losses.get(loss)

    @tf.function(
        input_signature=[
            [
                tf.TensorSpec(shape=[None, 1, 64], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
            ],
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        ],
    )
    def train_step(self, inputs, outputs):
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.loss_fn(outputs, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "matched_filters": keras.saving.serialize_keras_object(self.matched_filters),
                "trainable_conv": self.trainable_conv,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config["matched_filters"] = keras.saving.deserialize_keras_object(config["matched_filters"])

        return cls(**config)