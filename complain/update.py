import numpy as np
import pandas
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import getopt, sys
import pickle
from mtranslate import translate

def prepare_data(col_name, inputfile):
    # Read in data from CSV into a dataframe
    df = pandas.read_csv(open(inputfile, encoding='latin-1'), delimiter="|")

    # Clean up the data (remove NaNs, ensure labels are identical, etc.)
    df = df[pandas.notnull(df['Answer'])]
    df = df[pandas.notnull(df[col_name])]
    df[col_name] = df[col_name].apply(lambda x: x.title())
    # Translate
    print('Translating, this might take a while...')
    df['Answer'] = df['Answer'].apply(lambda x: translate(x, 'en'))

    df.to_csv('cleantranslated.csv', sep='|', encoding='latin-1')

    # Applying bag of words
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    counts = count_vect.fit_transform(df['Answer'])
    tfidf = tfidf_transformer.fit_transform(counts)

    # Split data into train-test
    x_train, x_test, y_train, y_test = train_test_split(tfidf, df[col_name], test_size=0.20, random_state=42)

    return x_train, x_test, y_train, y_test


def fit_model(clf, x_train, x_test, y_train, y_test):
    # Fit model
    clf.fit(x_train, y_train)
    # Predict data for evaluation
    predicted = clf.predict(x_test)
    # Print out some evaluations
    print("Accuracy: " + str(accuracy_score(y_test, predicted)))

    # Some other evaluations:

    # print("Avg Precision: " + str(average_precision_score(y_test, predicted)))
    # print("F1: " + str(f1_score(y_test, predicted)))
    #
    # tp, fp, tn, fn = confusion_matrix(y_test, predicted).ravel()
    # print("Size " + str(len(predicted)))
    # print("TP: " + str(tp) + " - FP: " + str(fp) + " - TN: " + str(tn) + " - FN: " + str(fn))

    # print(y_test)
    # print(predicted)

    return clf


def gridsearch(name, x_train, x_test, y_train, y_test):
    # Optimizes the hyperparameters for each possible model we choose
    # (we're just testing things here, once we figure out the best possible model and hyperparameters
    # we won't used gridsearch anymore)

    # Defining the search space for all the possible hyperparameters, for the models we want to try out

    # svm.SVC()
    parametersSVC = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1.0]}

    # svm.LinearSVC()
    parametersLinSVC = [
        {'C': [1, 10, 100], 'loss': ['hinge', 'squared_hinge'], 'penalty': ['l2'], 'dual': [True]},
        {'C': [1, 10, 100], 'loss': ['squared_hinge'], 'penalty': ['l1', 'l2'], 'dual': [False]}
    ]

    # This is how we would create a model with certain parameters once we've figured out the best one with gridsearch
    # clf = svm.NuSVC(gamma=1.0, kernel='rbf', nu=0.1)

    # Start GridSearch with given model and hyperparameters
    grid = GridSearchCV(svm.SVC(), parametersSVC)
    grid1 = GridSearchCV(svm.LinearSVC(), parametersLinSVC)

    # Start gridsearch. Gridsearch will try every model, with every possible combination of given hyperparameters
    # and evaluate each combination. Gridsearch uses cross validation for evaluation
    print("Fitting SVC")
    grid.fit(x_train, y_train)
    predicted = grid.predict(x_test)
    print("Fitting LinearSVC")
    grid1.fit(x_train, y_train)
    predicted1 = grid1.predict(x_test)

    print("===========================")
    print("Scores for: " + name)
    print("SVC")
    print(grid.best_params_)
    print(np.mean(predicted == y_test))
    print("LinearSVC")
    print(grid1.best_params_)
    print(np.mean(predicted1 == y_test))


def parse_args(argv):
    message = 'Usage: update.py -i <inputfile>'
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:")
    except getopt.GetoptError:
        print(message)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(message)
            sys.exit()
        elif opt == '-i':
            inputfile = arg
        else:
            print(message)
            sys.exit()

    if not inputfile:
        print(message)
        sys.exit(2)

    return inputfile


def update(inputfile, name, clf):
    print('Preparing data...')
    x_train, x_test, y_train, y_test = prepare_data(name.title(), inputfile)

    print('Encoding labels...')
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    pickle.dump(le, open('labelencoder_' + name + '.pickle', 'wb'))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    print('Fitting model...')
    model = fit_model(clf, x_train, x_test, y_train, y_test)
    pickle.dump(model, open('model_' + name + '.pickle', 'wb'))

    print('Done for ' + name)


def run_gridsearch(inputfile, name):
    print('Preparing data...')
    x_train, x_test, y_train, y_test = prepare_data(name.title(), inputfile)

    print('Encoding labels...')
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    print('Starting gridsearch for ' + name + ' ...')
    gridsearch(name, x_train, x_test, y_train, y_test)


def start():
    #inputfile = parse_args(sys.argv[1:])

    # run_gridsearch('training_data.csv', 'category')
    # run_gridsearch('training_data.csv', 'functionality')
    #
    # clf_category = svm.NuSVC(gamma=1.0, kernel='rbf', nu=0.01)
    clf_category = svm.LinearSVC(C=1, dual=False, loss='squared_hinge', penalty='l1')
    update('training_data.csv', 'category', clf_category)
    # #
    # clf_functionality = svm.LinearSVC(dual=True, loss='hinge', penalty='l2', C=1)
    # update('training_data.csv', 'functionality', clf_functionality)


if __name__ == "__main__":
    start()