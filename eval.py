from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from match import predict

from util import flat_read, map_item


path_test = 'data/test.csv'
texts = flat_read(path_test, 'text')
labels = flat_read(path_test, 'label')

label_set = sorted(list(set(labels)))

class_num = len(label_set)

paths = {'rank': 'metric/rank.csv',
         'freq': 'metric/freq.csv'}


def test(name, texts, labels, thre):
    preds = list()
    for text, label in zip(texts, labels):
        pred = predict(text, name, thre)
        preds.append(pred)
    precs = precision_score(labels, preds, average=None, labels=label_set)
    recs = recall_score(labels, preds, average=None, labels=label_set)
    with open(map_item(name, paths), 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(class_num):
            f.write('%s,%.2f,%.2f\n' % (label_set[i], precs[i], recs[i]))
    f1 = f1_score(labels, preds, average='weighted')
    print('\n%s f1: %.2f - acc: %.2f\n' % (name, f1, accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    test('rank', texts, labels, thre=0.2)
    test('freq', texts, labels, thre=0.2)
