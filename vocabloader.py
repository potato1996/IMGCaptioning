from build_vocab import Vocabulary


def vocab_loader(data_loc):
    """
    :param data_loc: the address of corpus
    :return: Vocabulary class variable vocab
    """
    with open(data_loc, 'r') as f:
        words = f.read().split("\n")
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    return vocab

# TEST
# vocab = get_loader("vocab.txt")
# print(len(vocab.word2idx.keys()))
# 9957
