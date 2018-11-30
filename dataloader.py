from build_vocab import Vocabulary


def get_loader(data_loc):
    """
    :param data_loc: the address of corpus
    :return: Vocabulary class variable vocab
    """
    with open(data_loc, 'r') as f:
        words = f.read().split()
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    return vocab

# TEST
# vocab = get_loader("vocab.txt")
# print(list(vocab.word2idx.keys())[:10])
# ['<pad>', '<start>', '<end>', '<unk>', 'a', 'very', 'clean', 'and', 'well', 'decorated']
