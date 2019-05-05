import os

import re
import jieba

from random import shuffle, randint

from util import load_word_re, load_type_re, load_pair, word_replace


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_cut_word = 'dict/cut_word.txt'
jieba.load_userdict(path_cut_word)

aug_rate = 2


def save_train(path, texts, labels):
    label_texts = dict()
    for text, label in zip(texts, labels):
        if label not in label_texts:
            label_texts[label] = list()
        cut_text = ' '.join(jieba.cut(text))
        label_texts[label].append(cut_text)
    head = 'label,cut_doc'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for label, texts in label_texts.items():
            f.write(label + ',' + ' '.join(texts) + '\n')


def save_test(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def gather(path_aug_dir, path_train, path_test):
    texts, labels = list(), list()
    files = os.listdir(path_aug_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_aug_dir, file), 'r') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(label)
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    texts, labels = zip(*texts_labels)
    bound = int(len(texts) * 0.9)
    save_train(path_train, texts[:bound], labels[:bound])
    save_test(path_test, texts[bound:], labels[bound:])


def clean(text):
    text = re.sub(stop_word_re, '', text)
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    return word_replace(text, syno_dict)


def augment(text, name):
    aug_texts = list()
    bound = len(text) - 1
    if bound > 0:
        for _ in range(aug_rate):
            words = list(text)
            if name == 'drop':
                ind = randint(0, bound)
                words.pop(ind)
            elif name == 'swap':
                ind1, ind2 = randint(0, bound), randint(0, bound)
                words[ind1], words[ind2] = words[ind2], words[ind1]
            elif name == 'copy':
                ind1, ind2 = randint(0, bound), randint(0, bound)
                words.insert(ind1, words[ind2])
            else:
                raise KeyError
            aug_texts.append(''.join(words))
    return aug_texts


def prepare(path_univ_dir, path_aug_dir):
    files = os.listdir(path_univ_dir)
    for file in files:
        text_set = set()
        texts = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = line.strip().lower()
                text = clean(text)
                if text and text not in text_set:
                    text_set.add(text)
                    texts.append(text)
                    texts.extend(augment(text, 'drop'))
                    texts.extend(augment(text, 'swap'))
                    texts.extend(augment(text, 'copy'))
        with open(os.path.join(path_aug_dir, file), 'w') as f:
            for text in texts:
                f.write(text + '\n')


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    path_aug_dir = 'data/aug'
    prepare(path_univ_dir, path_aug_dir)
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    gather(path_aug_dir, path_train, path_test)
