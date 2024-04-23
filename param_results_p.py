import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score


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
train_folders = ['/home/ayeung_umass_edu/nv-nets/results/250-ring-dataset-eps-0.40-p-{}/train'.format(i) for i in ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']]
test_folders = ['/home/ayeung_umass_edu/nv-nets/results/250-ring-dataset-eps-0.40-p-{}/test'.format(i) for i in ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']]


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

# Load all train and test datasets
all_train_datasets = {}
all_test_datasets = {}

for train_folder, test_folder in zip(train_folders, test_folders):
    p = extract_p_from_folder(train_folder)
    train_datasets = load_datasets(train_folder)
    test_datasets = load_datasets(test_folder)

    all_train_datasets[p] = train_datasets
    all_test_datasets[p] = test_datasets


"""
Get our predictions for each, record accuracy and precision scores
"""
mean_accuracies_train = {}
stdev_accuracies_train = {}

mean_precisions_train = {}
stdev_precisions_train = {}

mean_accuracies_test = {}
stdev_accuracies_test = {}

mean_precisions_test = {}
stdev_precisions_test = {}

for key in all_train_datasets.keys():
    accuracys_train = []
    precisions_train = []

    accuracys_test = []
    precisions_test = []

    for seed in all_train_datasets[key].keys():
        train_X = all_train_datasets[key][seed].f.arr_0
        test_X = all_test_datasets[key][seed].f.arr_0

        classifier = MNISTClassifier()
        classifier.learn(train_X, train_y)

        predictions = np.array(classifier.predict(train_X))

        accuracy = accuracy_score(train_y, predictions)
        accuracys_train.append(accuracy)


        precision = precision_score(train_y, predictions, average = 'weighted')
        precisions_train.append(precision)

        predictions = np.array(classifier.predict(test_X))

        accuracy = accuracy_score(test_y, predictions)
        accuracys_test.append(accuracy)


        precision = precision_score(test_y, predictions, average = 'weighted')
        precisions_test.append(precision)

    mean_accuracies_train[key] = np.mean(accuracys_train)
    stdev_accuracies_train[key] = np.std(accuracys_train)

    mean_accuracies_test[key] = np.mean(accuracys_test)
    stdev_accuracies_test[key] = np.std(accuracys_test)

    mean_precisions_train[key] = np.mean(precisions_train)
    stdev_precisions_train[key] = np.std(precisions_train)

    mean_precisions_test[key] = np.mean(precisions_test)
    stdev_precisions_test[key] = np.std(precisions_test)

ring_sizes = list(all_train_datasets.keys())

np.save('/home/ayeung_umass_edu/nv-nets/results/mean_accuracies_train_p.npy', mean_accuracies_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/mean_accuracies_test_p.npy', mean_accuracies_test, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/mean_precisions_train_p.npy', mean_precisions_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/mean_precisions_test_p.npy', mean_precisions_test, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_accuracies_train_p.npy', stdev_accuracies_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_accuracies_test_p.npy', stdev_accuracies_test, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_precisions_train_p.npy', stdev_precisions_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_precisions_test_p.npy', stdev_precisions_test, allow_pickle= True)
