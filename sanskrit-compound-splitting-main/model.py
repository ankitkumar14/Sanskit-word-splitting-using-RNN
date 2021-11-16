import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(
        self,
        batch_size,
        vocab_size,
        embed_dim,
        rnn_units,
        dropout_probability,
        filter_sizes,
        num_filters,
        max_length,
        num_classes,
    ):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.rnn_units = rnn_units
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.max_length = max_length
        self.num_classes = num_classes

        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(rnn_units, return_sequences=True), merge_mode="sum"
        )
        self.dropout = tf.keras.layers.Dropout(dropout_probability)
        self.convolutions = [
            tf.keras.layers.Conv2D(
                num_filters,
                (size, rnn_units),
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=0.0, stddev=0.1
                ),
            )
            for size in filter_sizes
        ]
        self.classification_layer = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        embedded_inputs = self.embedding(x)
        inputs = tf.nn.tanh(embedded_inputs)
        lstm_outputs = self.lstm(inputs)
        lstm_outputs_dropout = self.dropout(lstm_outputs)
        lstm_outputs_expanded = tf.expand_dims(lstm_outputs_dropout, -1)

        convolved_outputs = []
        for size, conv in zip(self.filter_sizes, self.convolutions):
            w = size // 2
            padded_lstm_outputs = tf.pad(
                lstm_outputs_expanded, [[0, 0], [w, w], [0, 0], [0, 0]], "CONSTANT"
            )
            conv_output = conv(padded_lstm_outputs)
            convolved_outputs.append(
                tf.reshape(conv_output, [-1, self.max_length, self.num_filters])
            )

        convolved_outputs = tf.concat(convolved_outputs, 2)
        outputs = tf.concat([convolved_outputs, lstm_outputs, inputs], axis=2)
        outputs = self.classification_layer(outputs)

        return outputs
