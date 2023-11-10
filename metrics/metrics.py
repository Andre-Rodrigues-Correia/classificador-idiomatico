import pickle

from utils.utils import Utils
from features.features import Features
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report
from nltk.classify.util import accuracy
import matplotlib.pyplot as plt
import seaborn as sns


class Metrics:

    @classmethod
    def get_extract_features(cls, feature_type, text):
        if feature_type == 'lemmatization':
            return Features.lemmatization_extract_features(text=text)
        if feature_type == 'recognize_cities':
            return Features.recognize_names_features(text=text)
        if feature_type == 'lemmatization_and_recognize_cities':
            return Features.lemmatization_and_recognize_names_features(text=text)
        else:
            return Features.basic_extract_features(text=text)

    @classmethod
    def get_dataset_type(cls, dataset_type=None):
        if dataset_type == 'random':
            return Utils.get_random_test_dataset()

        return Utils.get_test_dataset()

    @classmethod
    def get_accuracy(cls, model_type, feature_type, dataset_type=None):
        load_training = open(f'models/{model_type}_model_{feature_type}.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = cls.get_dataset_type(dataset_type=dataset_type)
        test_features = []
        for text, lang in test_data:
            features = cls.get_extract_features(feature_type=feature_type, text=text)
            test_features.append((features, lang))

        accuracy_score = accuracy(classifier, test_features)
        return accuracy_score

    @classmethod
    def evaluate_model(cls, model_type, feature_type, dataset_type=None):
        load_training = open(f'models/{model_type}_model_{feature_type}.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = cls.get_dataset_type(dataset_type=dataset_type)
        test_features = []
        for text, lang in test_data:
            features = cls.get_extract_features(feature_type=feature_type, text=text)
            test_features.append((features, lang))

        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        accuracy_score = accuracy(classifier, test_features)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        confusion = confusion_matrix(true_labels, predicted_labels)
        classification_rep = classification_report(true_labels, predicted_labels, target_names=classifier.labels())

        print("Accuracy:", accuracy_score)
        print("F1 Score:", f1)
        print("Recall:", recall)
        print("Precision:", precision)
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", classification_rep)

    @classmethod
    def export_evaluate_model(cls, model_type, feature_type, output_filename, dataset_type=None):
        load_training = open(f'models/{model_type}_model_{feature_type}.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = cls.get_dataset_type(dataset_type=dataset_type)
        test_features = []
        for text, lang in test_data:
            features = cls.get_extract_features(feature_type=feature_type, text=text)
            test_features.append((features, lang))

        accuracy_score = accuracy(classifier, test_features)

        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        confusion = confusion_matrix(true_labels, predicted_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classifier.labels(), yticklabels=classifier.labels())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(
            f'Accuracy: {accuracy_score}\n'
            f'F1 Score: {f1}\n'
            f'Precision: {precision}\n'
            f'Recall: {recall}\n'
            f'Confusion Matrix'
        )
        plt.savefig(f'datasets/validation/exported/{output_filename}', bbox_inches='tight')
        plt.close()
        print("Accuracy default dataset: ", accuracy_score)
