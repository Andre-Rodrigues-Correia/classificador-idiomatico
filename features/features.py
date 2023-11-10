from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.classify.util import accuracy
import spacy
import re
nlp = spacy.load("pt_core_news_sm")


class Features:

    @classmethod
    def basic_extract_features(cls, text):
        tokens = word_tokenize(text.lower())
        words = [token for token in tokens if token.isalpha()]
        feature = {word: True for word in words}
        return feature

    @classmethod
    def lemmatization_extract_features(cls, text):
        doc = nlp(text.lower())
        legitimatized_tokens = [token.lemma_ for token in doc]
        words = [token for token in legitimatized_tokens if token.isalpha()]
        feature = {word: True for word in words}
        return feature

    @classmethod
    def recognize_names_features(cls, text, names_brasil=None, names_portugal=None):
        if not names_brasil or not names_portugal:
            print('aqui')
            with open("datasets/cities_and_gentiles/brazil_names.txt", encoding="utf8") as cities_brasil_file:
                names_brasil = set(city.strip() for city in cities_brasil_file)

            with open("datasets/cities_and_gentiles/portugal_names.txt", encoding="utf8") as cities_portugal_file:
                names_portugal = set(city.strip() for city in cities_portugal_file)

        tokens = word_tokenize(text.lower())
        words = [token for token in tokens if token.isalpha()]
        feature = {word: True for word in words}

        # Identify multi-word city names using regular expressions
        brasil_city_names = []
        portugal_city_names = []

        for name in names_brasil:
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                brasil_city_names.append(match.group())

        for name in names_portugal:
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                portugal_city_names.append(match.group())

        for name in brasil_city_names:
            feature[f"brasil_city_{name}"] = 2  # Increase weight for city names from Brazil

        for name in portugal_city_names:
            feature[f"portugal_city_{name}"] = 2  # Increase weight for city names from Portugal

        return feature

    @classmethod
    def lemmatization_and_recognize_names_features(cls, text, names_brasil, names_portugal):
        # with open("datasets/cities_and_gentiles/brazil_names.txt", encoding="utf8") as cities_brasil_file:
        #     names_brasil = set(city.strip() for city in cities_brasil_file)
        #
        # with open("datasets/cities_and_gentiles/portugal_names.txt", encoding="utf8") as cities_portugal_file:
        #     names_portugal = set(city.strip() for city in cities_portugal_file)

        doc = nlp(text.lower())
        legitimatized_tokens = [token.lemma_ for token in doc]
        words = [token for token in legitimatized_tokens if token.isalpha()]
        feature = {word: True for word in words}

        brasil_city_names = []
        portugal_city_names = []

        for name in names_brasil:
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                brasil_city_names.append(match.group())

        for name in names_portugal:
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                portugal_city_names.append(match.group())

        for name in brasil_city_names:
            feature[f"brasil_city_{name}"] = 2  # Increase weight for city names from Brazil

        for name in portugal_city_names:
            feature[f"portugal_city_{name}"] = 2  # Increase weight for city names from Portugal

        print(feature)
        return feature
