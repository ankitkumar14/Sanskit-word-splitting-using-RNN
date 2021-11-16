import os
import json
import re
import random
import sys
import tensorflow as tf
from model import Model
from encoding import unicode_to_internal, internal_to_unicode

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

seed_value = 12
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

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

INPUT_TOKENIZER_PATH = os.path.join("tokenizers", "input_tokenizer.json")
TARGET_TOKENIZER_PATH = os.path.join("tokenizers", "target_tokenizer.json")

with open(INPUT_TOKENIZER_PATH) as f:
    data = json.load(f)
    input_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

with open(TARGET_TOKENIZER_PATH) as f:
    data = json.load(f)
    target_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)


INPUT_VOCAB_SIZE = len(input_tokenizer.word_index) + 1
TARGET_VOCAB_SIZE = len(target_tokenizer.word_index) + 1

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

optimizer = tf.keras.optimizers.Adam(learning_rate=LERANING_RATE)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
print(latest)
model.load_weights(latest).expect_partial()


def predict(sentence):

    inputs = [input_tokenizer.word_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=MAX_LENGTH, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = []

    predictions = model(inputs)

    for t in predictions[0]:
        predicted_id = tf.argmax(t).numpy()

        if predicted_id == 0:
            return "".join(result)

        result.append(target_tokenizer.index_word[predicted_id])
        decoder_input = tf.expand_dims([predicted_id], 0)

    result = "".join(result)
    return result


SYM_BOL = "^"
SYM_IDENT = "."
SYM_SPLIT = "="
SYM_DIVIDE = "-"
SYM_SPACE = "_"


def translate(input_sentence):
    sentence = re.sub(r" *[।\|\.,/\\'\"\?;:!\-] *", " ", input_sentence)
    sentence = unicode_to_internal(sentence)
    sentence = SYM_BOL + sentence

    prediction = predict(sentence)

    result = ""

    print(sentence)

    i = 1
    j = 1
    while i < len(sentence) and j < len(prediction):
        input_character, predicted_symbol = sentence[i], prediction[j]

        if prediction[j] == SYM_DIVIDE:
            if prediction[j + 1] == SYM_IDENT or prediction[j + 1] == SYM_SPLIT:
                result += prediction[j]
                print(prediction[j], "")
                j += 1
            else:
                result += prediction[j] + prediction[j + 1]
                print(prediction[j] + prediction[j + 1], "")
                j += 2
            continue

        if predicted_symbol == SYM_IDENT:
            result += input_character
        elif predicted_symbol == SYM_SPLIT:
            result += input_character + SYM_DIVIDE
        else:
            result += predicted_symbol

        print(predicted_symbol, input_character)
        i += 1
        j += 1

    print(result)

    result = result.replace("- ", " ").replace("= ", " ").replace("-", " ")
    result = internal_to_unicode(result)

    print("Input:", input_sentence)
    print("Prediction:", prediction)
    print("Result:", result)


if __name__ == "__main__":

    while True:
        sentence = input("Enter sentence: ")
        translate(sentence)
