import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import glob
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
print("Running, please wait.")

spamLabel = 1
hamLabel = 0


def removeStopWords(email):
    sw_english = stopwords.words('english')
    tokens = word_tokenize(email)
    cleaned_tokens = []
    for word in tokens:
        if word not in sw_english:
            cleaned_tokens.append(word)
    return " ".join(cleaned_tokens)


def get_data_list(folder_name):
    email_collection = []  # Store list of emails
    email_label = []  # Store corresponding labels
    # Storing OS independent paths for spam and ham , to be used during training.
    spamPath = os.path.join(folder_name, 'Spam') + os.sep
    hamPath = os.path.join(folder_name, 'Ham') + os.sep
    for folder_category in [hamPath, spamPath]:  # Running loop for different label of emails
        for file in glob.glob(folder_category + "*.txt"):  # iterate over files in a folder
            fileObject = open(file, "r+", encoding='utf-8', errors='ignore')  # open the file specified by a path
            email = fileObject.read()  # read in the file containing the email and store the string.
            email = email.lower()  # convert into lower case
            email = removeStopWords(email)  # remove stop words
            email_collection.append(email)  # add a email string to the list
            if "Ham" in folder_category:  # depending on the folder we are reading the email from label them
                email_label.append(hamLabel)
            elif "Spam" in folder_category:
                email_label.append(spamLabel)
    return np.array(email_collection), np.array(email_label)  # return email array and corresponding labels


def accuracy(predicted,
             original):  # take in predicted result as a list and match them with real labels to predict accuracy
    inaccuracies = 0  # number of inaccurate predictions
    for p, o in zip(predicted, original):  # zip merges predicted list and original list elements, iterate over them
        if p != o:  # count inaccuracies
            inaccuracies = inaccuracies + 1
    return 1 - inaccuracies / float(len(predicted))  # return accuracy % divided by 100.


x_train, y_train = get_data_list('Train')  # Get data and label for training purpose
x_test, y_test = get_data_list('TrainAccuracy')  # Get data and label for cross validation purpose
score = 0  # Store % inaccuracy divided by 100
maxScore = 0  # Store maximum score achieved after cross validation
maxMod = None  # Model corresponding to maximum possible cross validation score


def score_model(model_object, count_vectorizer):  # this method performs cross validation and returns accuracy% / 100
    test = count_vectorizer.transform(x_test)  # extract feature for cross validation data
    accuracy_factor = accuracy(model_object.predict(test), y_test)  # send for getting accuracy % /100
    return accuracy_factor


cv = CountVectorizer()  # extract feature, here frequency of a word.
features = cv.fit_transform(x_train)  # feature extraction of training data
model = svm.SVC(kernel='linear').fit(features, y_train)  # training SVM model with linear Kernel
curAcc = score_model(model_object=model, count_vectorizer=cv)  # cross validate
print("Linear Kernel Accuracy:" + str(curAcc))
if maxScore < curAcc:  # comparing accuracy with best model and store if necessary
    maxScore = curAcc
    maxMod = model
for C in range(1, 4):  # training SVM model with rbf Kernel with C=1,2,3
    model = svm.SVC(kernel='rbf', C=C).fit(features, y_train)  # train
    curAcc = score_model(model_object=model, count_vectorizer=cv)  # cross validate
    if maxScore < curAcc:  # comparing accuracy with best model and store if necessary
        maxScore = curAcc
        maxMod = model
    print("RBF Kernel with C=" + str(C) + " Accuracy :" + str(curAcc))
for degree in range(2, 4):  # training SVM model with polynomial Kernel with degree=2,3
    model = svm.SVC(kernel='poly', degree=degree).fit(features, y_train)
    curAcc = score_model(model_object=model, count_vectorizer=cv)  # cross validate
    if maxScore < curAcc:  # comparing accuracy with best model and store if necessary
        maxScore = curAcc
        maxMod = model
    print("Polynomial Kernel, degree=" + str(degree) + " Accuracy:" + str(curAcc))

filename_model = 'model.object'
pickle.dump(maxMod, open(filename_model, 'wb'))  # store best model's  object to disk
filename_cv = 'count_vectorizer.object'
pickle.dump(cv, open(filename_cv, 'wb'))  # store feature extractor to disk
