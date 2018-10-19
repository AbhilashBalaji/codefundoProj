import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


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
    model.add(Dense(8, input_dim=5, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


cols = ['Lat', 'Lon', 'pressure', 'wavedirection', 'waveheight']
pred = ['WindDir_L']
le = LabelEncoder()
le.fit(df['WindDir_L'])
df['WindDir_L'] = le.transform(df['WindDir_L'])
NBmodel(cols,pred)


x_cols = np.array(df[cols].values)
y_cols = np.array(df[pred].values)
y_cols = y_cols
x_train, x_test, y_train, y_test = train_test_split(
    x_cols, y_cols, test_size=0.33)
dummy_y=np_utils.to_categorical(y_cols)

estimator = KerasClassifier(build_fn=baseline_model,
                            epochs=200, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, x_cols, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" %
      (results.mean() * 100, results.std() * 100))
