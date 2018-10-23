import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout   
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD


df = pd.read_csv('IRENECSV.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df.columns)
# print(df[df['hurrthreat']==1].head)


def NBmodel(cols, pred):

    x_cols = np.array(df[cols].values)
    y_cols = np.array(df[pred].values)
    print(x_cols.shape)
    clf = GaussianNB()
    x_train, x_test, y_train, y_test = train_test_split(
        x_cols, y_cols, test_size=0.33)

    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=5))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])


cols = ['Lat', 'Lon','DaysTH', 'pressure', 'wavedirection', 'waveheight']
df=df[df['DaysTH'] [0]!='-']
df['DaysTH']=[i[:3] for i in df['DaysTH']]

pred = ['WindDir_L']
le = LabelEncoder()
le.fit(df['WindDir_L'])
df['WindDir_L'] = le.transform(df['WindDir_L'])
NBmodel(cols, pred)


x_cols = np.array(df[cols].values)
y_cols = np.array(df[pred].values)
y_cols = y_cols
x_train, x_test, y_train, y_test = train_test_split(
    x_cols, y_cols, test_size=0.33)
dummy_y = np_utils.to_categorical(y_cols)

estimator = KerasClassifier(build_fn=baseline_model,
                            epochs=2, batch_size=1, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, x_cols, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" %
      (results.mean() * 100, results.std() * 100))
