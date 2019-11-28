import nltk
# nltk.download('punkt')
import pickle
import argparse
from .util import load_file
from collections import Counter
import string



class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(anno_file, threshold):
    """Build a simple vocabulary wrapper."""

    annos = load_file(anno_file)
    counter = Counter()
    table = str.maketrans('-_', '  ')
    for vrelation in annos:
        relation = vrelation[4].translate(table)
        tokens = nltk.tokenize.word_tokenize(relation.lower())
        counter.update(tokens)


    # for word, cnt in counter.items():
    #     print(word, cnt)
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]


    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(args.caption_path, args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='../dataset/vidvrd/vrelation_train.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='../dataset/vidvrd/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=1,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)