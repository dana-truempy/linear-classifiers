"""
@param sys.argv[1] The name of the file containing the data set to be divided into training/testing data
@param sys.argv[2] The name of the file to write out the naive Bayes model and classifications
@param sys.argv[3] The name of the file to write out the logistic regression model and classifications
This is a main method to call all of the other scripts, takes one of the five data sets for this assignment and calls 
naive Bayes and logistic regression using five-fold cross-validation
"""

from csv import writer
import random
import sys

from CrossValidation import kFoldCrossValidation
import cancerProcess
import glassProcess
import irisProcess
import logistic
import naiveBayes
import soybeanProcess
import votesProcess

if 'iris' in sys.argv[1].lower():
    dataset = irisProcess.process(sys.argv[1])

elif 'cancer' in sys.argv[1].lower():
    dataset = cancerProcess.process(sys.argv[1])

elif 'votes' in sys.argv[1].lower():
    dataset = votesProcess.process(sys.argv[1])

elif 'glass' in sys.argv[1].lower():
    dataset = glassProcess.process(sys.argv[1])

elif 'soybean' in sys.argv[1].lower():
    dataset = soybeanProcess.process(sys.argv[1])

crossFolds = kFoldCrossValidation(dataset)
train = []
NBAccuracy = []  # get accuracy values for both tests
logAccuracy = []
NBWrite = []  # write out learned models and classifications for both tests in separate files
logWrite = []

for i in range(len(crossFolds)):
    train = []
    NBPredicts = []  # only want to write the last crossfold's predictions
    logPredicts = []

    NBPredicts.append(["Actual class", "Predicted class"])
    logPredicts.append(["Actual class", "Predicted class"])
    for crossFold in crossFolds[:i] + crossFolds[i + 1:]:  # use all crossfolds but the test one as training set
        train.extend(crossFold)

    classProbs, featureProbs = naiveBayes.NBTrain(train)
    line = []
    line.append(["Class probablities"])  # write the class probabilities for the trained model
    for key, value in classProbs.items():
        line.append("Class: " + str(key))
        line.append(value)
    NBWrite.append(line)

    line = []
    line.append(["Feature probabilities"])  # write the feature probabilities for the trained model
    for key, value in featureProbs.items():
        line.append("Class: " + str(key))
        for prob in value:
            line.append(prob)
    NBWrite.append(line)

    mistakes = 0  # now that the model has been created and recorded, test it
    for obs in crossFolds[i]:  # use other crossfold for testing
        prediction = naiveBayes.NBTest(obs, classProbs, featureProbs)
        NBPredicts.append([obs.classifier, prediction])
        if prediction != obs.classifier:  # if the nb classifies incorrectly, list as mistake
            mistakes += 1
    NBAccuracy.append((len(crossFolds[i]) - mistakes) / len(crossFolds[i]))  # get the accuracies of each cross-validation run

    numClasses, sortedData = logistic.classifyData(train)  # the classification counts how many classes are present and sorts the data into classes
    line = []
    line.append(["Learned model: "])
    if numClasses == 2:
        model = logistic.binaryLR(train)  # different method for two or more classes
        for value in model:  # binary only outputs one model to write
            line.append(value)
    else:
        model = logistic.multiLR(sortedData)
        for key, value in model.items():
            line.append("Class: " + str(key))
            for weight in value:  # multi outputs as many models as there are classes
                line.append(weight)

    logWrite.append(line)

    mistakes = 0
    for obs in crossFolds[i]:  # test LR the same way
        prediction = logistic.predictClass(obs, model, numClasses)
        logPredicts.append([obs.classifier, prediction])
        if prediction != obs.classifier:  # if the logistic regression classifies incorrectly, list as mistake
            mistakes += 1
    logAccuracy.append((len(crossFolds[i]) - mistakes) / len(crossFolds[i]))  # get the accuracies of each cross-validation run

NBWrite.append(["Accuracy for each of five crossfolds"])  # write out accuracy of each run
NBWrite.append(NBAccuracy)
NBWrite.append(NBPredicts)

logWrite.append(["Accuracy for each of five crossfolds"])
logWrite.append(logAccuracy)
logWrite.append(logPredicts)

print("Average accuracy over five-fold cross-validation:")
print("Naive Bayes: " + str(sum(NBAccuracy) / len(NBAccuracy)))  # print the average accuracy of the five runs
print("Logistic regression: " + str(sum(logAccuracy) / len(logAccuracy)))

with open(sys.argv[2], 'w', newline='') as NBFile, open(sys.argv[3], 'w', newline='') as logFile:
    writer1 = writer(NBFile)
    writer2 = writer(logFile)

    writer1.writerows(NBWrite)
    writer2.writerows(logWrite)
