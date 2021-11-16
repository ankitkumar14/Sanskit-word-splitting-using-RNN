import os
import random
import time

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

from data import DataLoader
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

seed_value = 12
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

DATA_DIR = "data"

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
VALIDATION_DATA_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

EMBEDDING_DIM = 128
RNN_UNITS = 256
DROPOUT_PROBABILITY = 0.2
FILTER_SIZES = [3, 5, 7]
NUM_FILTERS = 128

BATCH_SIZE = 32
EPOCHS = 10
LERANING_RATE = 0.005
MAX_LENGTH = 256

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch:03d}")

LOG_DIR = "logs"

validation_dataset = DataLoader(
    VALIDATION_DATA_PATH, max_length=MAX_LENGTH, batch_size=1
).get_dataset()

test_dataset = DataLoader(
    TEST_DATA_PATH, max_length=MAX_LENGTH, batch_size=1
).get_dataset()

train_data_loader = DataLoader(
    TRAIN_DATA_PATH,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    tokenizer_save_path="tokenizers",
)
train_dataset = train_data_loader.get_dataset()

INPUT_VOCAB_SIZE = train_data_loader.input_vocab_size
TARGET_VOCAB_SIZE = train_data_loader.target_vocab_size

model = Model(
    BATCH_SIZE,
    INPUT_VOCAB_SIZE,
    EMBEDDING_DIM,
    RNN_UNITS,
    DROPOUT_PROBABILITY,
    FILTER_SIZES,
    NUM_FILTERS,
    MAX_LENGTH,
    TARGET_VOCAB_SIZE,
)


class HammingDistance(tf.keras.metrics.Mean):
    def __init__(self, name="hamming_distance"):
        super().__init__(name=name)
        self._fn = self.hamming_distance
        self.__name__ = name

    def hamming_distance(self, y_true, y_pred):
        y_pred = tf.cast(
            tf.argmax(tf.nn.softmax(y_pred, axis=2), axis=2), dtype=y_true.dtype
        )
        result = tf.not_equal(y_true, y_pred)
        not_eq = tf.reduce_sum(tf.cast(result, tf.float32))
        ham_distance = tf.math.divide_no_nan(not_eq, result.shape[0])
        return ham_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        matches = self._fn(y_true, y_pred)
        return super().update_state(matches, sample_weight=sample_weight)


optimizer = tf.keras.optimizers.Adam(learning_rate=LERANING_RATE)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), HammingDistance()],
)

val_accuracy_metric = "val_sparse_categorical_accuracy"
val_loss_metric = "val_sparse_categorical_loss"

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor=val_accuracy_metric,
    save_best_only=True,
    save_weights_only=True,
    mode="max",
    verbose=1,
)

early_stopping_callback = EarlyStopping(
    monitor=val_accuracy_metric, mode="max", patience=4
)

reduce_lr_callcback = ReduceLROnPlateau(monitor=val_loss_metric, factor=0.2)

tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

callbacks = [
    checkpoint_callback,
    early_stopping_callback,
    reduce_lr_callcback,
    tensorboard_callback,
]

initial_epoch = 0

latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)

if latest:
    initial_epoch = int(latest.split("_")[1])
    latest = os.path.join(CHECKPOINT_DIR, latest)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()

history = model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_accuracy)
