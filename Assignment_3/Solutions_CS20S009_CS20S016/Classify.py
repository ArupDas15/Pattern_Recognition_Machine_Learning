import os
import glob
import pickle


def get_data_list(folder_name):
    email_collection = []  # store list of emails to be tested
    fileList = os.listdir(folder_name + os.sep)  # store list of files
    numberOfFiles = len(fileList)  # get length
    for i in range(1, numberOfFiles + 1):  # read files serially
        for file in glob.glob(folder_name + os.sep + "email" + str(i) + ".txt"):  # get that particular file path
            fileObject = open(file, "r+", encoding='utf-8', errors='ignore')  # open the file
            email = fileObject.read()  # read the file
            email = email.lower()  # convert to lower case
            email_collection.append(email)  # append the email to a list
    return email_collection


def classify():
    x_test = get_data_list('test')  # get list of emails in the test folder
    filename_model = 'model.object'
    filename_cv = 'count_vectorizer.object'
    if os.path.isfile(filename_model) and os.path.isfile(filename_cv):
        loaded_model = pickle.load(open(filename_model, 'rb'))  # load the SVM classifier
        cv_model = pickle.load(open(filename_cv, 'rb'))  # load the feature extractor
        test = cv_model.transform(x_test)  # extract the features, here frequency of words
        print(loaded_model.predict(test))  # prediction :D
        with open("output.txt", "w") as output:  # the output to a file
            output.write(str(loaded_model.predict(test)))
    else:
        print('Please copy '+filename_model+' and '+filename_cv+' files from zip into this directory before '
                                                                'running Classify.py ')


classify()
