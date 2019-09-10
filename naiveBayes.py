from copy import copy
import math
import operator
import sys


# This is a rewritten, less horrible version of my previous NB implementations from the first and second assignments
def NBTrain(train):
    """
    @param train The training data set
    @return classProbs, featureProbs The proportion of each class in the data and of each feature value in each class
    Finds the prior probabilities and outputs these values as the trained model.
    """
    classCounts = {}  # count the members of each class
    for obs in train:
        if obs.classifier not in classCounts:  # if the class is not yet in the dictionary, add it and count the first observation
            classCounts[obs.classifier] = 1
        else:
            classCounts[obs.classifier] += 1

    classProbs = classCounts.copy()  # now take the counts and divide them by the total to get class probabilities
    for classifier, value in classProbs.items():
        classProbs[classifier] = value / len(train)

    featureProbs = classProbs.copy()  # take the dictionary and replace the values with feature probabilities

    for className in featureProbs:
        featureProbs[className] = [1 for feat in range(len(train[0].features))]  # make probs have at least one observation so none are 0

    for line in train:  # go through the training data and count how many observations in each class have each feature value
        for index, feat in enumerate(line.features):
            if feat._value == 1:
                featureProbs[line.classifier][index] += 1

    for className, features in featureProbs.items():  # then divide the number with that feature over the total in the class
        for index, feat in enumerate(features):
            features[index] = feat / classCounts[className]

    return classProbs, featureProbs


def NBTest(obs, classProbs, featureProbs):
    """
    @param obs A single observation to be classified
    @param classProbs The class probabilities P(C)
    @param featureProbs The feature probabilities P(f|C)
    @return classPrediction The name of the class with the highest probability given the model
    This method takes an observation and the trained model and classifies the observation 
    """
    classPrediction = 0
    NBProbs = {}
    for className in classProbs:  # start with 1 for each since the values are multiplied together
        NBProbs[className] = 1

    for index, feature in enumerate(obs.features):
        if feature._value == 1:  # for each feature present, multiply by the prob of that feature in the data
            for name, prob in NBProbs.items():
                NBProbs[name] = prob * featureProbs[name][index]
        else:
            for name, prob in NBProbs.items():  # if feature is not present, multiply by 1-prob
                NBProbs[name] = prob * (1 - featureProbs[name][index])

    for name, prob in NBProbs.items():
        NBProbs[name] *= classProbs[name]  # multiply by the class probability

    classPrediction = max(NBProbs.items(), key=operator.itemgetter(1))[0]
    return(classPrediction)
