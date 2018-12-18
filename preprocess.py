import os

import re

from util import load_word_re, load_type_re, load_pair, word_replace


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)


def save(path, label_texts):
    head = 'label,doc'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for label, texts in label_texts.items():
            f.write(label + ',' + ' '.join(texts) + '\n')


def prepare(path, path_dir):
    label_texts = dict()
    files = os.listdir(path_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        text_set = set()
        label_texts[label] = list()
        with open(os.path.join(path_dir, file), 'r') as f:
            for line in f:
                text = re.sub(stop_word_re, '', line.strip())
                for word_type, word_re in word_type_re.items():
                    text = re.sub(word_re, word_type, text)
                text = word_replace(text, homo_dict)
                text = word_replace(text, syno_dict)
                if text not in text_set:
                    text_set.add(text)
                    label_texts[label].append(text)
    save(path, label_texts)


if __name__ == '__main__':
    path = 'data/train.csv'
    path_dir = 'data/train'
    prepare(path, path_dir)
