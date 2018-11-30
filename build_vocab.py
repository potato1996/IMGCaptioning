import nltk
from collections import Counter
from pycocotools.coco import COCO


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

    # as you mentioned
    def size(self):
        return len(self.word2idx)

    def word2vec(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    # 'unk' or 'pad' this is a question
    def vec2word(self, vec):
        if not vec in self.idx2word:
            return '<unk>'
            # return 'pad'
        return self.idx2word[vec]


def build_vocab(json, threshold=5):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for id in ids:
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

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
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'w') as f:
        # example: for mapping {"a":1, "b":2} create and write "a b" into vocab.path
        f.write(" ".join(list(vocab.word2idx.keys())))
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Coco annotation dataset address
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation file')
    # Address to store the mapping
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.txt',
                        help='path for saving vocabulary wrapper')
    # Optional argument, 4 or 5 or 10, try it
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
