import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from keras.datasets import mnist
from sklearn.metrics import accuracy_score, precision_score


"""
Load MNIST Data (We will only use train_y and test_y)
"""
(train_X, train_y), (test_X, test_y) = mnist.load_data()

"""
Returns a RidgeClassifier learned for classifying the digit i based on X and y
Note: i must be from 0-9
"""

def make_classifier(i, X, y):
    classifier = RidgeClassifier(alpha = 1e-4)
    y_i = (y == i).astype(int)
    classifier.fit(X, y_i)
    return classifier

"""
Class for our own MNIST classifier based on an ensemble of these RidgeClassifiers
"""
class MNISTClassifier():
    def __init__(self):
        self.classifiers = []
    
    #Makes 10 classifiers for each digit 0-9 and appends them to the classifiers list
    def learn(self, train_X, train_y):
        for i in range(10):
            classifier = make_classifier(i, train_X, train_y)
            self.classifiers.append(classifier)
        return
    
    def predict_instance(self, x):
        x = x.reshape(1, -1)
        maxval = -np.inf
        maxi = 0
        for i in range(len(self.classifiers)):
            if self.classifiers[i].decision_function(x) > maxval:
                maxval = self.classifiers[i].decision_function(x)
                maxi = i
        return maxi
      
    def predict(self, X):
        predictions = list(map(self.predict_instance, X))
        return predictions
    

"""
Load data
"""
param_range = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']

train_folders = ['/work/pi_erietman_umass_edu/ayeung_umass_edu/300-ring-dataset-eps-{}-p-{}/train'.format(i,j) for i in param_range for j in param_range]
test_folders = ['/work/pi_erietman_umass_edu/ayeung_umass_edu/300-ring-dataset-eps-{}-p-{}/test'.format(i,j) for i in param_range for j in param_range]


def extract_seed(filename):
    return int(filename.split('_')[-1].split('rings')[-1].split('.')[0])  # Extract seed from filename

def extract_ring_size_from_folder(folder):
    return int(folder.split('/')[-2].split('-')[0])

def extract_eps_from_folder(folder):
    return float(folder.split('/')[-2].split('-')[-3])

def extract_p_from_folder(folder):
    return float(folder.split('/')[-2].split('-')[-3])

# Function to load datasets from a folder
def load_datasets(folder):
    datasets = {}
    for filename in os.listdir(folder):
        if filename.endswith('.npz'):
            seed = extract_seed(filename)
            datasets[seed] = np.load(os.path.join(folder, filename))
    return datasets

train_accuracies = {}
train_precisions = {}

test_accuracies = {}
test_precisions = {}

for train_folder, test_folder in zip(train_folders, test_folders):
    eps = extract_eps_from_folder(train_folder)
    train_accuracies[eps] = {}
    train_precisions[eps] = {}

    test_accuracies[eps] = {}
    test_precisions[eps] = {}

    p = extract_p_from_folder(train_folder)

    train_datasets = load_datasets(train_folder)
    test_datasets = load_datasets(test_folder)

    accuracies1 = []
    precisions1= []

    accuracies2 = []
    precisions2 = []
    for seed in train_datasets.keys():
        train_X = train_datasets[seed].f.arr_0
        test_X = test_datasets[seed].f.arr_0

        classifier = MNISTClassifier()
        classifier.learn(train_X, train_y)

        train_predictions = np.array(classifier.predict(train_X))

        train_accuracy = accuracy_score(train_y, train_predictions)
        accuracies1.append(train_accuracy)

        train_precision = precision_score(train_y, train_predictions, average= 'weighted')
        precisions1.append(train_precision)

        test_predictions = np.array(classifier.predict(test_X))

        test_accuracy = accuracy_score(test_y, test_predictions)
        accuracies2.append(test_accuracy)

        test_precision = precision_score(test_y, test_predictions, average= 'weighted')
        precisions2.append(test_precision)

    train_accuracies[eps][p] = np.mean(accuracies1)
    train_precisions[eps][p] = np.mean(precisions1)

    test_accuracies[eps][p] = np.mean(accuracies2)
    test_precisions[eps][p] = np.mean(precisions2)


np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/train_accuracies.npy', train_accuracies, allow_pickle= True)
np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/train_precisions.npy', train_precisions, allow_pickle= True)
np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/test_accuracies.npy', test_accuracies, allow_pickle= True)
np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/test_precisions.npy', test_precisions, allow_pickle= True)