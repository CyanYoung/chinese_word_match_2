import numpy as np

from collections import Counter

import matplotlib.pyplot as plt

from util import flat_read


path_vocab_freq = 'stat/vocab_freq.csv'
path_len_freq = 'stat/len_freq.csv'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial Unicode MS']


def count(path_freq, items, field):
    pairs = Counter(items)
    sort_items = [item for item, freq in pairs.most_common()]
    sort_freqs = [freq for item, freq in pairs.most_common()]
    with open(path_freq, 'w') as f:
        f.write('item,freq' + '\n')
        for item, freq in zip(sort_items, sort_freqs):
            f.write(str(item) + ',' + str(freq) + '\n')
    plot_freq(sort_items, sort_freqs, field, u_bound=20)


def plot_freq(items, freqs, field, u_bound):
    inds = np.arange(len(items))
    plt.bar(inds[:u_bound], freqs[:u_bound], width=0.5)
    plt.xlabel(field)
    plt.ylabel('freq')
    plt.xticks(inds[:u_bound], items[:u_bound], rotation='vertical')
    plt.show()


def statistic(path_train):
    docs = flat_read(path_train, 'doc')
    all_words = ' '.join(docs).split()
    doc_words = [doc.split() for doc in docs]
    doc_lens = [len(words) for words in doc_words]
    count(path_vocab_freq, all_words, 'vocab')
    count(path_len_freq, doc_lens, 'doc_len')


if __name__ == '__main__':
    path_train = 'data/train.csv'
    statistic(path_train)
