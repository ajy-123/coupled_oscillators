import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
from sklearn.model_selection import train_test_split


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
train_folders = ['/work/pi_erietman_umass_edu/ayeung_umass_edu/Diff_Ring_Sizes/{}-ring-dataset/train'.format(i) for i in range(50, 501, 50)]
test_folders = ['/work/pi_erietman_umass_edu/ayeung_umass_edu/Diff_Ring_Sizes/{}-ring-dataset/test'.format(i) for i in range(50, 501, 50)]


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
# Function to stratify and subsample the dataset
def subsample_dataset(X, y, test_size_fraction):
    # Use stratified sampling to preserve label distribution
    X_subsampled, _, y_subsampled, _ = train_test_split(X, y, train_size=test_size_fraction, stratify=y, random_state=42)
    return X_subsampled, y_subsampled

# Subsample sizes based on MNIST test set size
test_size = test_X.shape[0]
subsample_fractions = [0.25, 0.5, 0.75, 1.0]

"""
Get our predictions for each, record accuracy and precision scores
"""
mean_accuracies_train = {}
stdev_accuracies_train = {}

mean_accuracies_test = {}
stdev_accuracies_test = {}

# Adjust the loop to handle subsampling
for size in all_train_datasets.keys():
    mean_accuracies_train[size] = []
    stdev_accuracies_train[size] = []
    
    mean_accuracies_test[size] = []
    stdev_accuracies_test[size] = []
    
    for fraction in subsample_fractions:
        accuracys_train = []
        accuracys_test = []

        for seed in all_train_datasets[size].keys():
            train_X = all_train_datasets[size][seed].f.arr_0
            test_X = all_test_datasets[size][seed].f.arr_0

            # Subsample the training dataset
            subsample_size = int(test_size * fraction)
            train_X_subsampled, train_y_subsampled = subsample_dataset(train_X, train_y, subsample_size / len(train_X))

            classifier = MNISTClassifier()
            classifier.learn(train_X_subsampled, train_y_subsampled)

            # Training accuracy
            predictions_train = np.array(classifier.predict(train_X_subsampled))
            accuracy_train = accuracy_score(train_y_subsampled, predictions_train)
            accuracys_train.append(accuracy_train)

            # Test accuracy
            predictions_test = np.array(classifier.predict(test_X))
            accuracy_test = accuracy_score(test_y, predictions_test)
            accuracys_test.append(accuracy_test)

        # Save mean and standard deviation for the current fraction and ring size
        mean_accuracies_train[size].append(np.mean(accuracys_train))
        stdev_accuracies_train[size].append(np.std(accuracys_train))

        mean_accuracies_test[size].append(np.mean(accuracys_test))
        stdev_accuracies_test[size].append(np.std(accuracys_test))

# Save results as .npy files with more descriptive names
np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/train_accuracy_means_by_subsample_fraction.npy', mean_accuracies_train, allow_pickle=True)
np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/train_accuracy_stdev_by_subsample_fraction.npy', stdev_accuracies_train, allow_pickle=True)
np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/test_accuracy_means_by_subsample_fraction.npy', mean_accuracies_test, allow_pickle=True)
np.save('/work/pi_erietman_umass_edu/ayeung_umass_edu/test_accuracy_stdev_by_subsample_fraction.npy', stdev_accuracies_test, allow_pickle=True)

"""
Create plots to show how training and test accuracy change with different subsample fractions for each ring size
"""
for size in all_train_datasets.keys():
    # Create a plot for training and test accuracy vs subsample fraction
    plt.errorbar(subsample_fractions, mean_accuracies_train[size], yerr=stdev_accuracies_train[size], label="Train Accuracy", fmt='-o')
    plt.errorbar(subsample_fractions, mean_accuracies_test[size], yerr=stdev_accuracies_test[size], label="Test Accuracy", fmt='-x')

    plt.xlabel("Subsample fraction of training set")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Subsample fraction for Ring size {size}")
    plt.legend()
    
    # Save plot with descriptive filename
    plt.savefig(f'/work/pi_erietman_umass_edu/ayeung_umass_edu/accuracy_vs_subsample_ring_size_{size}.png')
    plt.clf()