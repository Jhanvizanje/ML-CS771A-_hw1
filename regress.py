import os
import numpy as np

def load_data(file_path):
    return np.load(file_path, allow_pickle=True, encoding='latin1')

def calculate_class_means(X_train_data):
    return [np.mean(X_train_data[class_idx], axis=0) for class_idx in range(40)]

def train_linear_regression(Seen_Classes_Attributes, class_means, regularization_lambda):
    class_similarity_matrix = np.dot(Seen_Classes_Attributes.T, Seen_Classes_Attributes)
    identity_matrix = np.eye(85)
    weighted_class_means = np.dot(Seen_Classes_Attributes.T, class_means)
    learned_weights = np.dot(np.linalg.inv(class_similarity_matrix + (regularization_lambda * identity_matrix)), weighted_class_means)
    return learned_weights

def predict_classes(X_test_data, learned_weights, Unseen_Classes_Attributes):
    class_models = [np.dot(learned_weights.T, attributes) for attributes in Unseen_Classes_Attributes]
    predicted_class = []
    for X_sample in X_test_data:
        distances = [np.linalg.norm(model - X_sample) for model in class_models]
        predicted_class.append(distances.index(min(distances)) + 1)
    return predicted_class

x_mean=0
def calculate_mean(X_TEST):
    for x_temp in X_TEST:
        x_mean+=x_temp
    x_mean= x_mean/len(X_TEST)
    return x_mean

def calculate_accuracy(Y_test_labels, predicted_class_labels):
    correct_predictions = sum(1 for true, predicted in zip(Y_test_labels, predicted_class_labels) if true == predicted)
    return correct_predictions / len(predicted_class_labels)

def dataset_path(fileName):
    current_directory = os.path.dirname(__file__)  # Get the current directory of your script
    data_file_path = os.path.join(current_directory, 'Data_Set', fileName)
    return data_file_path

if __name__ == "__main__":
    # Load data with relative paths

    X_TRAIN = load_data(dataset_path('X_seen.npy'))
    X_TEST = load_data(dataset_path('Xtest.npy'))
    Y_TEST = load_data(dataset_path('Ytest.npy'))

    # Load class attributes with relative paths
    Seen_Classes_Attributes = load_data(dataset_path('class_attributes_seen.npy'))
    UnSeen_Classes_Attributes = load_data(dataset_path('class_attributes_unseen.npy'))

    class_means = calculate_class_means(X_TRAIN)
    
    regularization_values = [0.01, 0.1, 1, 10, 20, 50, 100]
    for regularization_lambda in regularization_values:
        learned_weights = train_linear_regression(Seen_Classes_Attributes, class_means, regularization_lambda)
        predicted_class_labels = predict_classes(X_TEST, learned_weights, UnSeen_Classes_Attributes)
        accuracy = calculate_accuracy(Y_TEST, predicted_class_labels)
        print(f"Regularization Lambda = {regularization_lambda}, Accuracy = {accuracy}")
