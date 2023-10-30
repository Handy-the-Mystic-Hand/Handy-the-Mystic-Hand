import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])

labels = np.asarray(data_dict['labels'])

x_, x_test, t_, t_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

x_train, x_val, t_train, t_val = train_test_split(x_, t_, test_size=0.2, shuffle=True, stratify=t_)

model = RandomForestClassifier()
model.fit(x_train, t_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, t_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()