import io
import os
import json
import tensorflow as tf

INPUT_START_TOKEN = "^"
TARGET_START_TOKEN = "."


def preprocess_example(l):
    x, y = l.split("\t")

    x = INPUT_START_TOKEN + x
    x = " ".join(list(x))

    y = TARGET_START_TOKEN + y
    y = " ".join(list(y))

    return x, y


class DataLoader:
    def __init__(
        self,
        data_path,
        max_length=128,
        batch_size=1,
        split_ratio=0.1,
        tokenizer_save_path=None,
    ):

        self.max_length = max_length
        self.tokenizer_save_path = tokenizer_save_path

        (
            input_tensor,
            target_tensor,
            input_tokenizer,
            target_tokenizer,
        ) = self.load_dataset(data_path)

        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer

        self.input_vocab_size = len(input_tokenizer.word_index) + 1
        self.target_vocab_size = len(target_tokenizer.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor, target_tensor)
        ).shuffle(len(input_tensor))

        self.dataset = dataset.batch(batch_size, drop_remainder=True)

        if tokenizer_save_path:
            self.save_tokenizer(tokenizer_save_path)

    def get_examples(self, path):
        lines = io.open(path, encoding="UTF-8").read().strip().split("\n")
        word_pairs = [preprocess_example(l) for l in lines]
        input_sequence = [word_pair[0] for word_pair in word_pairs]
        target_sequence = [word_pair[1] for word_pair in word_pairs]
        return input_sequence, target_sequence

    def load_dataset(self, path):
        input_sequence, target_sequence = self.get_examples(path)

        input_tensor, input_tokenizer = self.tokenize(input_sequence)
        target_tensor, target_tokenizer = self.tokenize(target_sequence)

        return (
            input_tensor,
            target_tensor,
            input_tokenizer,
            target_tokenizer,
        )

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            split=" ", filters="", oov_token="?", lower=False
        )
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(
            tensor, padding="post", maxlen=self.max_length
        )
        return tensor, lang_tokenizer

    def get_dataset(self):
        return self.dataset

    def save_tokenizer(self, save_path):
        input_tokenizer = self.input_tokenizer.to_json()
        target_tokenizer = self.target_tokenizer.to_json()

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(f"{save_path}/input_tokenizer.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(input_tokenizer, ensure_ascii=False))

        with open(f"{save_path}/target_tokenizer.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(target_tokenizer, ensure_ascii=False))
