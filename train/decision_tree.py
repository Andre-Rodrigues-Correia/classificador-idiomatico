import pickle

from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
from features.features import Features


class DecisionTree:
    INDEX_SENTENCE = 1
    INDEX_LANGUAGE = 0

    @classmethod
    def train_model(cls, feature_type=None):
        train_set = []

        with open("datasets/mid_corpus.txt", encoding="utf8") as read_file:
            for sentence in read_file:
                splited_sentences = sentence.split('\t')
                train_set.append((splited_sentences[cls.INDEX_SENTENCE], splited_sentences[cls.INDEX_LANGUAGE]))

        training_features = []
        if feature_type == 'lemmatization':
            for text, lang in train_set:
                feature = Features.lemmatization_extract_features(text)
                training_features.append((feature, lang))
        if feature_type == 'recognize_names':
            for text, lang in train_set:
                feature = Features.recognize_names_features(text)
                training_features.append((feature, lang))
        if feature_type == 'lemmatization_and_recognize_names':
            for text, lang in train_set:
                feature = Features.lemmatization_and_recognize_names_features(text)
                training_features.append((feature, lang))
        else:
            for text, lang in train_set:
                feature = Features.basic_extract_features(text)
                training_features.append((feature, lang))

        classifier = DecisionTreeClassifier.train(training_features)

        accuracy_score = accuracy(classifier, training_features)
        print("Classifier accuracy:", accuracy_score)

        save_training = open(f'models/decision_tree_model_{feature_type}.pickle', 'wb')
        pickle.dump(classifier, save_training)
        save_training.close()
