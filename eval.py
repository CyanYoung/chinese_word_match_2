from sklearn.metrics import accuracy_score, f1_score

from match import predict

from util import flat_read


path_test = 'data/test.csv'
texts = flat_read(path_test, 'text')
labels = flat_read(path_test, 'label')


def test(name, texts, labels, thre):
    preds = list()
    for text, label in zip(texts, labels):
        pred = predict(text, name, thre)
        preds.append(pred)
    f1 = f1_score(labels, preds, average='weighted')
    print('\n%s f1: %.2f - acc: %.2f\n' % (name, f1, accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    test('rank', texts, labels, thre=0.2)
    test('freq', texts, labels, thre=0.2)
