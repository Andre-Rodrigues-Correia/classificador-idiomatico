import pandas as pd
from matplotlib import pyplot as plt

from train.naive_bayes import NaiveBayes
from classifier.classfier import Classifier
from metrics.metrics import Metrics
from train.maximum_entropy import MaxEnt
from train.decision_tree import DecisionTree
from utils.utils import Utils

if __name__ == '__main__':
    feature_types = {
        0: 'basic',  # é ralizada apenas a tokenização do texto
        1: 'lemmatization',
        2: 'recognize_names',
        3: 'lemmatization_and_recognize_names'
    }

    model_types = {
        0: 'naive_bayes',
        1: 'max_ent',
        2: 'decision_tree'
    }

    for feature in range(3):
        NaiveBayes.train_model(feature_type=feature_types.get(feature))
        MaxEnt.train_model(feature_type=feature_types.get(feature))
        DecisionTree.train_model(feature_type=feature_types.get(feature))





