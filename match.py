import json

import re

import jieba

import numpy as np

from util import load_word_re, load_type_re, load_pair, word_replace, map_item


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_cut_word = 'dict/cut_word.txt'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
jieba.load_userdict(path_cut_word)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_rank = 'feat/rank.json'
path_freq = 'feat/freq.json'
with open(path_rank, 'r') as f:
    rank_dict = json.load(f)
with open(path_freq, 'r') as f:
    freq_dict = json.load(f)

feats = {'rank': rank_dict,
         'freq': freq_dict}


def predict(text, name, max_cand):
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
    words = list(jieba.cut(text))
    label_pairs = map_item(name, feats)
    labels = list(label_pairs.keys())
    scores = list()
    for pairs in label_pairs.values():
        match_scores = list()
        for word in words:
            if word in pairs:
                match_scores.append(pairs[word])
        if match_scores:
            scores.append(sum(match_scores) / len(match_scores))
        else:
            scores.append(0.0)
    scores = np.array(scores)
    bound = min(len(scores), max_cand)
    max_scores = sorted(scores, reverse=True)[:bound]
    max_inds = np.argsort(-scores)[:bound]
    max_preds = [labels[ind] for ind in max_inds]
    formats = list()
    for pred, score in zip(max_preds, max_scores):
        formats.append('{} {:.3f}'.format(pred, score))
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('rank: %s' % predict(text, 'rank', max_cand=5))
        print('freq: %s' % predict(text, 'freq', max_cand=5))
