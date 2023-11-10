import pickle

from features.features import Features


class Classifier:

    @classmethod
    def get_extract_features(cls, feature_type, text):
        if feature_type == 'lemmatization':
            return Features.lemmatization_extract_features(text=text)
        if feature_type == 'recognize_names':
            return Features.recognize_names_features(text=text)
        if feature_type == 'lemmatization_and_recognize_names':
            return Features.lemmatization_and_recognize_names_features(text=text)
        else:
            return Features.basic_extract_features(text=text)

    @classmethod
    def classifier_text(cls, text, model_type, feature_type=None):
        try:
            load_training = open(f'models/{model_type}_model_{feature_type}.pickle', 'rb')
            classifier = pickle.load(load_training)
            features = cls.get_extract_features(feature_type=feature_type, text=text)
            predicted_lang = classifier.classify(features)
            probabilities = classifier.prob_classify(features)
            probability_correct = probabilities.prob(predicted_lang)
            return predicted_lang, probability_correct
        except:
            return None
