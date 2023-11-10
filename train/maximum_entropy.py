import pickle
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from features.features import Features
from concurrent.futures import ProcessPoolExecutor
import itertools


class MaxEnt:
    INDEX_SENTENCE = 1
    INDEX_LANGUAGE = 0

    @classmethod
    def train_model(cls, feature_type=None):
        train_set = []

        # Carregar nomes do Brasil e Portugal uma única vez
        names_brasil = set()
        names_portugal = set()
        with open("datasets/cities_and_gentiles/brazil_names.txt", encoding="utf8") as cities_brasil_file:
            names_brasil = set(city.strip() for city in cities_brasil_file)
        with open("datasets/cities_and_gentiles/portugal_names.txt", encoding="utf8") as cities_portugal_file:
            names_portugal = set(city.strip() for city in cities_portugal_file)

        # Leitura do arquivo em lotes
        batch_size = 1000
        with open("datasets/big_corpus.txt", encoding="utf8") as read_file:
            batch = list(itertools.islice(read_file, batch_size))
            while batch:
                with ProcessPoolExecutor() as executor:
                    train_set += list(executor.map(cls.process_line, batch, itertools.repeat(feature_type),
                                                   itertools.repeat(names_brasil), itertools.repeat(names_portugal)))
                batch = list(itertools.islice(read_file, batch_size))

        # Treinamento do classificador
        classifier = MaxentClassifier.train(train_set)
        accuracy_score = accuracy(classifier, train_set)
        print("Classifier accuracy:", accuracy_score)

        save_training = open(f'models/max_ent_model_{feature_type}.pickle', 'wb')
        pickle.dump(classifier, save_training)
        save_training.close()

    @classmethod
    def process_line(cls, sentence, feature_type, names_brasil, names_portugal):
        splited_sentences = sentence.split('\t')
        text, lang = splited_sentences[cls.INDEX_SENTENCE], splited_sentences[cls.INDEX_LANGUAGE]

        if feature_type == 'lemmatization':
            feature = Features.lemmatization_extract_features(text)
        elif feature_type == 'recognize_names':
            feature = Features.recognize_names_features(text, names_brasil, names_portugal)
        elif feature_type == 'lemmatization_and_recognize_names':
            feature = Features.lemmatization_and_recognize_names_features(text, names_brasil, names_portugal)
        else:
            feature = Features.basic_extract_features(text)

        return (feature, lang)
