import json

import jieba

import numpy as np

from preprocess import clean

from util import map_item


path_rank = 'feat/rank.json'
path_freq = 'feat/freq.json'
with open(path_rank, 'r') as f:
    rank_dict = json.load(f)
with open(path_freq, 'r') as f:
    freq_dict = json.load(f)

feats = {'rank': rank_dict,
         'freq': freq_dict}


def predict(text, name, cand):
    text = clean(text)
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
    bound = min(len(scores), cand)
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
        print('rank: %s' % predict(text, 'rank', cand=5))
        print('freq: %s' % predict(text, 'freq', cand=5))
