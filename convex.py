import os
import numpy as np
from statistics import mean

def load_data(file_path):
    return np.load(file_path, allow_pickle=True, encoding='latin1')

def calculate_similarity_matrix(X_TRAIN, Seen_Classes_Attributes, UnSeen_Classes_Attributes):
    Similarity_Matrix = []
    Class_Means = [np.mean(X_TRAIN[i], axis=0) for i in range(40)]
    for i in range(10):
        similarity_scores = []
        for j in range(40):
            similarity_scores.append(np.dot(Seen_Classes_Attributes[j], UnSeen_Classes_Attributes[i]))
        Similarity_Matrix.append(similarity_scores)

    # Normalize the similarity scores
    for i in range(10):
        total_score = sum(Similarity_Matrix[i])
        Similarity_Matrix[i] = [score / total_score for score in Similarity_Matrix[i]]
    
    return Similarity_Matrix, Class_Means

def calculate_predicted_unseen_classes(Similarity_Matrix, Class_Means):
    Predicted_UnSeen_Classes = []
    for i in range(10):
        temp = [Similarity_Matrix[i][j] * Class_Means[j] for j in range(40)]
        temp1 = temp[0]
        for j in range(1, 40):
            temp1 = [temp1[k] + temp[j][k] for k in range(len(temp[0]))]
        Predicted_UnSeen_Classes.append(temp1)
    
    return Predicted_UnSeen_Classes

x_mean=0
def calculate_mean(X_TEST):
    for x_temp in X_TEST:
        x_mean+=x_temp
    x_mean= x_mean/len(X_TEST)
    return x_mean

def predict_classes(X_TEST, Predicted_UnSeen_Classes):
    Predicted_Classes = []
    for X in X_TEST:
        distance_to_unseen = [np.linalg.norm(attributes - X) for attributes in Predicted_UnSeen_Classes]
        predicted_class = distance_to_unseen.index(min(distance_to_unseen)) + 1
        Predicted_Classes.append(predicted_class)
    
    return Predicted_Classes

def calculate_accuracy(Y_TEST, Predicted_Classes):
    correct_predictions = sum(1 for true_class, predicted_class in zip(Y_TEST, Predicted_Classes) if true_class == predicted_class)
    accuracy = correct_predictions / len(Predicted_Classes)
    return accuracy

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

    Similarity_Matrix, Class_Means = calculate_similarity_matrix(X_TRAIN, Seen_Classes_Attributes, UnSeen_Classes_Attributes)
    Predicted_UnSeen_Classes = calculate_predicted_unseen_classes(Similarity_Matrix, Class_Means)
    Predicted_Classes = predict_classes(X_TEST, Predicted_UnSeen_Classes)
    accuracy = calculate_accuracy(Y_TEST, Predicted_Classes)

    print(f"Accuracy: {accuracy}")
