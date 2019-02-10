import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
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

def create_network():
    model = Sequential()
    model.add(Dense(10, input_dim=X.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Activation('relu'))
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    return model

# Load 2008 training data
X, y = load('./data/train_2008.csv')
y = keras.utils.np_utils.to_categorical(y)

neural_network = KerasClassifier(build_fn=create_network, epochs=10, batch_size=128, verbose=1)
scores = cross_val_score(neural_network, X, y, cv=3)
print(scores)
