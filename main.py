import pandas as pd
from flask import Flask
from flask_cors import CORS
from matplotlib import pyplot as plt
from flask import Flask, make_response, request
from train.naive_bayes import NaiveBayes
from classifier.classfier import Classifier
from metrics.metrics import Metrics
from train.maximum_entropy import MaxEnt
from train.decision_tree import DecisionTree
from utils.utils import Utils

app = Flask(__name__)


@app.route('/classifier', methods=["POST"])
def get_classification():
    content = request.get_json()
    text = content['text']
    predicted_lang, probability_correct = Classifier.classifier_text(text=text, model_type='max_ent',
                                                                     feature_type='recognize_names')
    response = {
        'text': text,
        'language': predicted_lang,
        'probability': probability_correct
    }
    return make_response(response, 200)


if __name__ == '__main__':
    CORS(app)
    app.run()
