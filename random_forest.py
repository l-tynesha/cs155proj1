import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def load(filename):
    print('Loading 2008 Training data...', end='')
    
    f = open(filename, 'rb')
    data = np.loadtxt(f, dtype='float', delimiter=',', skiprows=1)
    f.close()
    
    X, y = data[:, 1:-1], data[:, -1]
    X = np.asarray(X)
    y = y.astype('int')
    
    print('DONE')
    return X, y


X, y = load('./data/train_2008.csv')

clf = RandomForestClassifier(criterion='entropy', max_depth=10, max_features=None)
scores = cross_val_score(clf, X, y, cv=3)

print(scores)
print(np.average(scores))