import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data']).tolist()

labels = np.asarray(data_dict['labels'])

x_train, x_val, t_train, t_val = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# n_estimators = [100, 150, 200, 250]
# max_features = ["sqrt", "log2", None]
# criterions = ["gini", "entropy", "log_loss"]
# max_depth = [10, 20, 30, 40, 50]
# curr_score = 0
# curr_model = None
# for estimator in n_estimators:
#     for feature in max_features:
#         for criterion in criterions:
#             for depth in max_depth:
#                 model = RandomForestClassifier(n_estimators=estimator, criterion=criterion, max_depth=depth, max_features=feature)
#                 model.fit(x_train, t_train)
#                 y_predict = model.predict(x_val)
#                 score = accuracy_score(y_predict, t_val)
#                 if score > curr_score:
#                     curr_model = model
#                 if score > 95:
#                     break
# print('{}% of validation accuracy for n_estimators: {}, criterion: {}, max_depth: {}, min_samples_split: {}, max_feature: {}'.format(
#     accuracy_score(curr_model.predict(x_val), t_val) * 100, curr_model.n_estimators, curr_model.criterion,
#     curr_model.max_depth, curr_model.min_samples_split, curr_model.max_features))
curr_model = RandomForestClassifier()
curr_model.fit(x_train, t_train)
y_predict = curr_model.predict(x_val)
score = accuracy_score(y_predict, t_val)
print('{}% of validation accuracy for n_estimators: {}, criterion: {}, max_depth: {}, min_samples_split: {}, max_feature: {}'.format(
    accuracy_score(curr_model.predict(x_val), t_val) * 100, curr_model.n_estimators, curr_model.criterion,
    curr_model.max_depth, curr_model.min_samples_split, curr_model.max_features))
f = open('model.p', 'wb')
pickle.dump({'model': curr_model}, f)
f.close()