import getopt
import pickle
import sys

import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def parse_args(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print('predict.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('predict.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt == '-i':
            inputfile = arg
        elif opt == '-o':
            outputfile = arg
        else:
            print('predict.py -i <inputfile> -o <outputfile>')
            sys.exit()

    if not inputfile and not outputfile:
        print('predict.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    return inputfile, outputfile


def init_bow(df, count_vect, tfidf_transformer):
    counts = count_vect.fit_transform(df)
    tfidf_transformer.fit(counts)


def bag_of_words(df, count_vect, tfidf_transformer):
    # Vectorize the text using bag of words
    counts = count_vect.transform(df)
    tfidf = tfidf_transformer.transform(counts)

    return tfidf


def start(ifile, ofile):
    print('Preparing data...')
    orig_df = pandas.read_csv(open('training_data.csv', encoding='latin-1'), delimiter="|")
    orig_df = orig_df[pandas.notnull(orig_df['Answer'])]
    orig_df = orig_df[pandas.notnull(orig_df['Category'])]

    # TODO: bag of words should come from data to predict.. not training data...
    cv = CountVectorizer()
    tfidf_tf = TfidfTransformer()
    init_bow(orig_df['Answer'], cv, tfidf_tf)

    df = pandas.read_csv(open(ifile, encoding='latin-1'), delimiter="|")
    df = df[pandas.notnull(df['Answer'])]
    tfidf = bag_of_words(df['Answer'], cv, tfidf_tf)

    cat_model = pickle.load(open('model_category.pickle', 'rb'))
    fun_model = pickle.load(open('model_functionality.pickle', 'rb'))
    cat_le = pickle.load(open('labelencoder_category.pickle', 'rb'))
    fun_le = pickle.load(open('labelencoder_functionality.pickle', 'rb'))

    print('Predicting category...')
    cat_pred = cat_model.predict(tfidf)
    cat_labels = cat_le.inverse_transform(cat_pred)
    df['Category'] = cat_labels

    orig_df_fun = pandas.read_csv(open('training_data.csv', encoding='latin-1'), delimiter="|")
    orig_df_fun = orig_df_fun[pandas.notnull(orig_df_fun['Answer'])]
    orig_df_fun = orig_df_fun[pandas.notnull(orig_df_fun['Functionality'])]

    cv = CountVectorizer()
    tfidf_tf = TfidfTransformer()
    init_bow(orig_df_fun['Answer'], cv, tfidf_tf)

    tfidf = bag_of_words(df['Answer'], cv, tfidf_tf)

    print('Predicting functionality...')
    fun_pred = fun_model.predict(tfidf)
    fun_labels = fun_le.inverse_transform(fun_pred)
    df['Functionality'] = fun_labels

    print('Writing data...')
    df.to_csv(ofile, sep='|', index=False)

    print('Done.')


if __name__ == "__main__":
    inputfile, outputfile = parse_args(sys.argv[1:])
    start(inputfile, outputfile)



