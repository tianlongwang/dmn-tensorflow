# Common data loading utilities.
import pickle
import copy
import os
import numpy as np


class DataSet:
    def __init__(self, batch_size, xs, qs, ys,ls, fact_counts, shuffle=True, name="dataset"):
        assert batch_size <= len(xs), "batch size cannot be greater than data size."
        self.name = name
        self.xs = xs
        self.qs = qs
        self.ys = ys
        self.ls = ls
        self.fact_counts = fact_counts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = len(self.xs)
        self.setup()

    def setup(self):
        self.indexes = list(range(self.count))  # used in shuffling
        self.current_index = 0
        self.num_batches = int(self.count / self.batch_size)
        self.reset()

    def next_batch(self):
        assert self.has_next_batch(), "End of epoch. Call 'complete_epoch()' to reset."
        from_, to = self.current_index, self.current_index + self.batch_size
        cur_idxs = self.indexes[from_:to]
        xs, qs, ys, ls = zip(*[[self.xs[i], self.qs[i], self.ys[i], self.ls[i]] for i in cur_idxs])
        self.current_index += self.batch_size
        return xs, qs, ys, ls

    def has_next_batch(self):
        return self.current_index + self.batch_size  <= self.count

    def split_dataset(self, split_ratio):
        """ Splits a data set by split_ratio.
        (ex: split_ratio = 0.3 -> this set (70%) and splitted (30%))
        :param split_ratio: ratio of train data
        :return: val_set
        """
        end_index = int(self.count * (1. - split_ratio))
        print('split_ratio', split_ratio)
        print('self.count',self.count)
        print('end_index', end_index)


        # do not (deep) copy data - just modify index list!
        val_set = copy.copy(self)
        val_set.count = self.count - end_index
        print('val_set.count', val_set.count)
        print('batch_size', self.batch_size)
        if(val_set.count < self.batch_size):
            print("WARNING: known bug when valuation set length smaller than batch size")
        val_set.indexes = list(range(end_index, self.count))
        val_set.num_batches = int(val_set.count / val_set.batch_size )#when val smaller than a batch, fixed it
        print('val_set.num_batches', val_set.num_batches)
        self.count = end_index
        self.setup()
        return val_set

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)


class WordTable:
    def __init__(self, word2vec=None, embed_size=0):
        self.word2vec = word2vec
        self.word2idx = {}
        self.idx2word = ['<eos>']  # zero padding will be <eos>
        self.embed_size = embed_size

    def add_vocab(self, words):
        """ Add vocabularies to dictionary. """
        if type(words) == str:
            word = words
            if self.word2vec and (word not in self.word2vec):
                self._create_vector(word)

            if word not in self.word2idx:
                index = len(self.idx2word)
                self.word2idx[word] = index
                self.idx2word.append(word)

        if type(words) == list:
            for word in words:
                if self.word2vec and (word not in self.word2vec):
                    self._create_vector(word)

                if word not in self.word2idx:
                    index = len(self.idx2word)
                    self.word2idx[word] = index
                    self.idx2word.append(word)

    def vectorize(self, word):
        """ Converts word to vector.
        :param word: string
        :return: 1-D array (vector)
        """
        self.add_vocab(word)
        return self.word2vec[word]

    def _create_vector(self, word):
        # if the word is missing from Glove, create some fake vector and store in glove!
        vector = np.random.uniform(0.0, 1.0, (self.embed_size,))
        if word == '<eos>':
            vector = np.zeros(self.embed_size)
        self.word2vec[word] = vector
        print("create_vector => %s is missing" % word)
        return vector

    def word_to_index(self, word):
        self.add_vocab(word)
        return self.word2idx[word]

    def index_to_word(self, index):
        return self.idx2word[index]

    @property
    def vocab_size(self):
        return len(self.idx2word)


def load_glove(dim):
    """ Loads GloVe data.
    :param dim: word vector size (50, 100, 200)
    :return: GloVe word table
    """
    word2vec = {}

    path = "data/glove/glove.6B." + str(dim) + 'd'
    fn = path+ '.cache'
    if os.path.exists(fn) and os.stat(fn).st_size > 0:
        with open(fn, 'rb') as cache_file:
            word2vec = pickle.load(cache_file)

    else:
        # Load n create cache
        with open(path + '.txt') as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = [float(x) for x in l[1:]]

        with open(path + '.cache', 'wb') as cache_file:
            pickle.dump(word2vec, cache_file)

    print("Loaded Glove data")
    return word2vec
