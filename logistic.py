import math
import operator
import random

from CategoricalFeature import CategoricalFeature

eta = 0.01  # tunable parameter, set to 0.01 for all data sets because it did not significantly change output


def classifyData(data):
    """
    @param data A data set of Observation objects
    @return len(classData), classData The number of classes and a dictionary of the data set sorted by class
    This method takes a data set and divides it up by class
    """
    classData = {}
    for obs in data:
        obs.features.append(CategoricalFeature(1))  # append a "feature" to the end of each observation that will represent the intercept value in the learned model
        if obs.classifier not in classData:
            classData[obs.classifier] = []
        classData[obs.classifier].append(obs)  # sort the observations by their classifier

    if len(classData) >= 2:
        return len(classData), classData
    else:
        raise Exception("Data must contain two or more classes. The data contain {} classes.".format(len(classData)))


def sigmoid(value):
    """
    @param value The value with which to calculate the sigmoid function
    @return The output of the sigmoid function using the given value
    This method represents a sigmoid function
    """
    return (1 / (1 + math.exp(-value)))


def softmax(vector):
    """
    @param vector A list of values with which a weighted sigmoid function is generated
    @return softmax A vector of values that should be weighted toward 0 or 1, with the 1 being the max value
    This method takes a vector of dot products for each class and weights it to skew toward one maximum value
    """
    sum = 0
    for classifier, value in vector.items():
        sum += math.exp(value)

    softmax = {classifier:0 for classifier in vector}
    for classifier, value in vector.items():  # divide each e^value by the total sum of all e^values
        softmax[classifier] = math.exp(value) / sum

    return softmax


def binaryLR(data):
    """
    @param data The original data set (not classified)
    @return weight The list of weights for each feature (and the intercept) that constitutes the learned model
    This method generates a logistic regression model for a data set containing two classes
    """
    numFeatures = len(data[0].features)
    weight = [random.uniform(-0.01, 0.01) for index in range(numFeatures)]  # generate random small values for each starting weight
    for i in range(500):  # run an arbitary large number of times - in the future could implement a minimum delta weight and/or early stopping
        deltaWeight = [0 for weight in range(numFeatures)]  # initialize all the weight changes to 0
        for obs in data:
            weightSum = 0
            for index, feature in enumerate(obs.features):
                weightSum += weight[index] * feature._value  # the dot product of the weight vector and the feature vector
            sigValue = sigmoid(weightSum)
            if obs.classifier == 0:
                bernoulli = 1  # a value representing which class the observation belongs to
            else:
                bernoulli = 0
            for index, delta in enumerate(deltaWeight):
                deltaWeight[index] += (bernoulli - sigValue) * obs.features[index]._value
        for index, value in enumerate(weight):
            weight[index] += eta * deltaWeight[index]  # multiply the weight change by eta (global parameter) then change weights by that much

    for obs in data:
        obs.features.pop(-1)  # at the end of the training, remove the artificial feature used for the intercept

    return(weight)


def multiLR(data):
    """
    @param data The dictionary of data values divided into classes (output from classifyData)
    @return weight A dictionary of lists of weights for each feature (and the intercept) for each class that constitute the learned models
    This method generates a logistic regression model for a data set containing more than two classes
    """
    weight = {classifier:[] for classifier in data}  # this weight vector must be a dictionary since each class needs its own set of weights
    for classifier in data:
        weight[classifier] = [random.uniform(-0.01, 0.01) for index in range(len(data[classifier][0].features))]  # initialize with small values
    for i in range(500):
        deltaWeight = {classifier:[0 for index in range(len(data[classifier][0].features))] for classifier in data}  # initialize to 0
        for classifier, observations in data.items():
            for obs in observations:
                weightSum = {classifier:0 for classifier in data}  # a dot product with the feature values is found for each class weight vector
                for classifier in data:
                    for index, feature in enumerate(obs.features):
                        weightSum[classifier] += weight[classifier][index] * feature._value
                softMax = softmax(weightSum)  # use the softmax instead of sigmoid
                rValue = {classifier:[] for classifier in data}  # a value indicating whether an observation belongs to a particular class
                for classifier in data:
                    if obs.classifier == classifier:  # weight the model for correct predictions by giving the correct classifier a 1 and all others 0
                        rValue[classifier] = 1
                    else:
                        rValue[classifier] = 0
                for classifier, weights in deltaWeight.items():
                    for index, change in enumerate(weights):
                        deltaWeight[classifier][index] += (rValue[classifier] - softMax[classifier]) * obs.features[index]._value
        for classifier, weights in weight.items():
            for index, value in enumerate(weights):
                weight[classifier][index] += eta * deltaWeight[classifier][index]

    for classifier, values in data.items():
        for obs in values:
            obs.features.pop(-1)

    return(weight)


def predictClass(obs, model, numClasses):
    """
    @param obs An Observation from the training data to be classified
    @param model The list (or lists) of weights that make up the learned model
    @param numClasses How many classes are in the data set - determines how the model will be used
    @return The predicted class for the observation
    This method uses a learned logistic regression model to classify a novel sample and return this classification
    """
    obs.features.append(CategoricalFeature(1))
    if numClasses == 2:
        dot = 0
        for index, feature in enumerate(obs.features):
            dot += feature._value * model[index]
        prob = sigmoid(dot)
        obs.features.pop(-1)
        if prob > 0.5:
            return 0
        else:
            return 1
    else:
        dots = {classifier:0 for classifier in model}
        for classifier, weights in model.items():
            for index, weight in enumerate(weights):
                dots[classifier] += weight * obs.features[index]._value
            probs = softmax(dots)
        obs.features.pop(-1)
        return(max(probs.items(), key=operator.itemgetter(1))[0])

