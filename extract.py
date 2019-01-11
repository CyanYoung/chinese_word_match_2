import json
import pickle as pk

import jieba
from jieba.analyse import textrank

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from util import load_word, flat_read, make_dict


min_freq = 1
key_num = 10

path_not_key = 'dict/not_key.txt'
not_keys = load_word(path_not_key)
jieba.analyse.set_stop_words('dict/not_key.txt')

path_rank = 'feat/rank.json'
path_tfidf = 'model/tfidf.pkl'
path_freq = 'feat/freq.json'

pos_set = ('a', 'ad', 'an', 'd', 'f', 'm', 'n', 'nr', 'ns', 'nt', 'nz',
           'q', 't', 'v', 'vd', 'vn', 'x')


def ind2word(word_inds):
    ind_words = dict()
    for word, ind in word_inds.items():
        ind_words[ind] = word
    return ind_words


def rank_fit(docs, labels, path):
    label_pairs = dict()
    for doc, label in zip(docs, labels):
        pairs = textrank(doc, topK=key_num, withWeight=True, allowPOS=pos_set)
        pairs = make_dict(pairs)
        label_pairs[label] = pairs
    with open(path, 'w') as f:
        json.dump(label_pairs, f, ensure_ascii=False, indent=4)
    if __name__ == '__main__':
        print(label_pairs)


def freq_fit(cut_docs, labels, path_feat, path_model):
    model = TfidfVectorizer(token_pattern='\w+', min_df=min_freq, stop_words=not_keys)
    model.fit(cut_docs)
    vecs = model.transform(cut_docs).toarray()
    label_pairs = dict()
    word_inds = model.vocabulary_
    ind_words = ind2word(word_inds)
    for vec, label in zip(vecs, labels):
        max_scores = np.array(sorted(vec, reverse=True)[:key_num])
        max_inds = np.argsort(-vec)[:]
        keys = [ind_words[max_ind] for max_ind in max_inds]
        pairs = [(key, score) for key, score in zip(keys, max_scores)]
        pairs = make_dict(pairs)
        label_pairs[label] = pairs
    with open(path_feat, 'w') as f:
        json.dump(label_pairs, f, ensure_ascii=False, indent=4)
    with open(path_model, 'wb') as f:
        pk.dump(model, f)
    if __name__ == '__main__':
        print(label_pairs)


def fit(path_train):
    docs = flat_read(path_train, 'doc')
    labels = flat_read(path_train, 'label')
    cut_docs = list()
    for doc in docs:
        texts = doc.split()
        doc_words = list()
        for text in texts:
            text_words = list(jieba.cut(text))
            doc_words.extend(text_words)
        cut_docs.append(' '.join(doc_words))
    rank_fit(docs, labels, path_rank)
    freq_fit(cut_docs, labels, path_freq, path_tfidf)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    fit(path_train)
