import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset from a pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert the data and labels to lists for compatibility with the RandomForestClassifier
data = np.asarray(data_dict['data']).tolist()
labels = np.asarray(data_dict['labels'])

# Split the dataset into training and validation sets
x_train, x_val, t_train, t_val = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Hyperparameters to be tested in the grid search
n_estimators = [100, 150, 200, 250]
max_features = ["sqrt", "log2", None]
criterions = ["gini", "entropy", "log_loss"]
max_depth = [10, 20, 30, 40, 50]

# Variables to keep track of the best model and its score
curr_score = 0
curr_model = None

# Grid search to find the best hyperparameters
for estimator in n_estimators:
    for feature in max_features:
        for criterion in criterions:
            for depth in max_depth:
                # Initialize the model with the current set of hyperparameters
                model = RandomForestClassifier(n_estimators=estimator, criterion=criterion, max_depth=depth, max_features=feature)
                # Train the model on the training set
                model.fit(x_train, t_train)
                # Predict on the validation set
                y_predict = model.predict(x_val)
                # Calculate the accuracy score
                score = accuracy_score(y_predict, t_val)
                # Update the best model if the current one is better
                if score > curr_score:
                    curr_model = model
                # Optional early stopping if accuracy exceeds 95%
                if score > 95:
                    break

# Print the performance of the best model along with its hyperparameters
print('{}% of validation accuracy for n_estimators: {}, criterion: {}, max_depth: {}, min_samples_split: {}, max_feature: {}'.format(
    accuracy_score(curr_model.predict(x_val), t_val) * 100, curr_model.n_estimators, curr_model.criterion,
    curr_model.max_depth, curr_model.min_samples_split, curr_model.max_features))

# Save the best model to a pickle file for future use
f = open('model.p', 'wb')
pickle.dump({'model': curr_model}, f)
f.close()
