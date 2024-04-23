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
train_folders = ['/home/ayeung_umass_edu/nv-nets/results/{}-ring-dataset/train'.format(i) for i in range(50, 501, 50)]
test_folders = ['/home/ayeung_umass_edu/nv-nets/results/{}-ring-dataset/test'.format(i) for i in range(50, 501, 50)]


def extract_seed(filename):
    return int(filename.split('_')[-1].split('rings')[-1].split('.')[0])  # Extract seed from filename

def extract_ring_size_from_folder(folder):
    return int(folder.split('/')[-2].split('-')[0])

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
    ring_size = extract_ring_size_from_folder(train_folder)
    train_datasets = load_datasets(train_folder)
    test_datasets = load_datasets(test_folder)

    all_train_datasets[ring_size] = train_datasets
    all_test_datasets[ring_size] = test_datasets


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

for size in all_train_datasets.keys():
    accuracys_train = []
    precisions_train = []

    accuracys_test = []
    precisions_test = []

    for seed in all_train_datasets[size].keys():
        train_X = all_train_datasets[size][seed].f.arr_0
        test_X = all_test_datasets[size][seed].f.arr_0

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

    mean_accuracies_train[size] = np.mean(accuracys_train)
    stdev_accuracies_train[size] = np.std(accuracys_train)

    mean_accuracies_test[size] = np.mean(accuracys_test)
    stdev_accuracies_test[size] = np.std(accuracys_test)

    mean_precisions_train[size] = np.mean(precisions_train)
    stdev_precisions_train[size] = np.std(precisions_train)

    mean_precisions_test[size] = np.mean(precisions_test)
    stdev_precisions_test[size] = np.std(precisions_test)

ring_sizes = list(all_train_datasets.keys())

np.save('/home/ayeung_umass_edu/nv-nets/results/mean_accuracies_train.npy', mean_accuracies_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/mean_accuracies_test.npy', mean_accuracies_test, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/mean_precisions_train.npy', mean_precisions_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/mean_precisions_test.npy', mean_precisions_test, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_accuracies_train.npy', stdev_accuracies_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_accuracies_test.npy', stdev_accuracies_test, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_precisions_train.npy', stdev_precisions_train, allow_pickle= True)
np.save('/home/ayeung_umass_edu/nv-nets/results/stdev_precisions_test.npy', stdev_precisions_test, allow_pickle= True)

plt.errorbar(x = ring_sizes, y = list(mean_accuracies_train.values()), yerr= list(stdev_accuracies_train.values()))
plt.xlabel("Ring Size")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs. Ring Size")
plt.savefig('/home/ayeung_umass_edu/nv-nets/results/training_accuracy.png')
plt.clf()

plt.errorbar(x = ring_sizes, y = list(mean_accuracies_test.values()), yerr= list(stdev_accuracies_test.values()))
plt.xlabel("Ring Size")
plt.ylabel("Accuracy")
plt.title("Test Accuracy vs. Ring Size")
plt.savefig('/home/ayeung_umass_edu/nv-nets/results/test_accuracy.png')