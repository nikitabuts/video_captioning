from collections import Counter
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


def start_end(captions):
    for img_i in range(len(captions)):
        for caption_i in range(len(captions[img_i])):
            sentence = captions[img_i][caption_i]
            captions[img_i][caption_i] = ["#START#"]+sentence.split(' ')+["#END#"]


def build_vocab(captions):
    word_counts = Counter()
    for img_i in range(len(captions)):
        for caption_i in range(len(captions[img_i])):
            sentence = captions[img_i][caption_i][1:-1]
            word_num = Counter(sentence)
            for word in word_num:
                word_counts[word] += word_num[word]
    return word_counts


def pad(word_counts):
    vocab = ['#UNK#', '#START#', '#END#', '#PAD#']
    vocab += [k for k, v in word_counts.items() if v >= 5 if k not in vocab]
    n_tokens = len(vocab)

    word_to_index = {w: i for i, w in enumerate(vocab)}

    eos_ix = word_to_index['#END#']
    unk_ix = word_to_index['#UNK#']
    pad_ix = word_to_index['#PAD#']

    return word_to_index, vocab, eos_ix, unk_ix, pad_ix


def as_matrix(sequences, word_to_index, pad_ix, unk_ix, max_len=None):
    max_len = max_len or max(map(len, sequences))

    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
    for i, seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix


def generate_caption(image, vocab, word_to_index,
                     unk_ix, pad_ix,
                     inception, recurrent,
                     caption_prefix=("#START#",),
                     t=1, sample=True, max_len=100):
    image = plt.imread(image)
    image = Image.fromarray(image).resize((299, 299))
    image = np.array(image).astype('float32') / 255.0

    assert isinstance(image, np.ndarray) and np.max(image) <= 1 \
           and np.min(image) >= 0 and image.shape[-1] == 3

    with torch.no_grad():
        image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

        vectors_8x8, vectors_neck, logits = inception(image[None])
        caption_prefix = list(caption_prefix)

        for _ in range(max_len):

            prefix_ix = as_matrix([caption_prefix], word_to_index=word_to_index,
                                  pad_ix=pad_ix, unk_ix=unk_ix)
            prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64)
            next_word_logits = recurrent.forward(vectors_neck, prefix_ix)[0, -1]
            next_word_probs = F.softmax(next_word_logits, dim=-1).numpy()

            assert len(next_word_probs.shape) == 1, 'probs must be one-dimensional'
            next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t)  # apply temperature

            if sample:
                next_word = np.random.choice(vocab, p=next_word_probs)
            else:
                next_word = vocab[np.argmax(next_word_probs)]

            caption_prefix.append(next_word)

            if next_word == "#END#":
                break

    return caption_prefix
