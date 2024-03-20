from sentence_transformers import SentenceTransformer
import numpy as np
from dalpy import randomForestkNN



sBertModel = SentenceTransformer('distilbert-base-nli-mean-tokens')

classifier = randomForestkNN(np.array([[0, 0]]), np.array([0]), n_trees=0)
classifier.fromTxtFile("refFigClassifier")



def classifyRefFig(refFigs):
    """refFigs is a list of strings, each string refers to a figure, a reference, or nothing"""
    classifiedRefFigs = []
    for elt in refFigs:
        classifiedRefFigs.append(classifier.predict(sBertModel.encode(elt)))
    return classifiedRefFigs