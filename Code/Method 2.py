import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from matplotlib import pyplot as plt

'''
Read in data
'''


total_data = pd.read_csv("sweatshirt_total.csv", low_memory = False)
total_data = total_data.fillna(0)


stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer("[a-z']+")


def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [stemmer.stem(t) for t in tokens]


def get_tf(data, use_idf, max_df=1.0, min_df=1):
    if use_idf:
        m = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english',
                            tokenizer=tokenize)
    else:
        m = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english',
                            tokenizer=tokenize)

    d = m.fit_transform(data)
    return m, d



total_data['reviewText'] = [str (item) for item in total_data['reviewText']]
tfidf_tm, tfidf_td = get_tf(total_data['reviewText'], use_idf=True, max_df=0.90, min_df=10)


#function get_train_test() seperates dataset by categories
def get_train_test(tfidf_trans, dataset, cat, testsize):
    # random_state=42
    X_train, X_test, y_train, y_test = train_test_split(tfidf_trans, dataset[cat], test_size=testsize, random_state=42)
    return X_train, X_test, y_train, y_test


category = ['color', 'size', 'qualiti', 'comfort', 'price', 'materi']


# function prediction_with_tfidf() uses six classifiers to predict y_predict vector
def prediction_with_tfidf(X_1, y_1, X_2):
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_1, y_1)
    y_predict_dt = model_dt.predict(X_2)

    model_nb = MultinomialNB()
    model_nb.fit(X_1, y_1)
    y_predict_nb = model_nb.predict(X_2)

    model_sgd = SGDClassifier()
    model_sgd.fit(X_1, y_1)
    y_predict_sgd = model_sgd.predict(X_2)

    model_svc = SVC(random_state=42)
    model_svc.fit(X_1, y_1)
    y_predict_svc = model_svc.predict(X_2)

    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_1, y_1)
    y_predict_rf = model_rf.predict(X_2)

    return y_predict_dt, y_predict_nb, y_predict_sgd, y_predict_svc, y_predict_rf


# function get_pred_scores() provides accuracy, precision, recall and f1-score of y_predict and y_test
def get_pred_scores(X_train, X_test, y_train, y_test):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    y_predict_dt, y_predict_nb, y_predict_sgd, y_predict_svc, y_predict_rf = prediction_with_tfidf(X_train, y_train, X_test)

    # generate random y_predict to provide a baseline
    # np.random.shuffle(y_predict_dt)
    # np.random.shuffle(y_predict_nb)
    # np.random.shuffle(y_predict_sgd)
    # np.random.shuffle(y_predict_svc)
    # np.random.shuffle(y_predict_rf)

    accuracy.append(accuracy_score(y_test, y_predict_dt))
    accuracy.append(accuracy_score(y_test, y_predict_nb))
    accuracy.append(accuracy_score(y_test, y_predict_sgd))
    accuracy.append(accuracy_score(y_test, y_predict_svc))
    accuracy.append(accuracy_score(y_test, y_predict_rf))

    precision.append(precision_score(y_test, y_predict_dt, average='weighted'))
    precision.append(precision_score(y_test, y_predict_nb, average='weighted'))
    precision.append(precision_score(y_test, y_predict_sgd, average='weighted'))
    precision.append(precision_score(y_test, y_predict_svc, average='weighted'))
    precision.append(precision_score(y_test, y_predict_rf, average='weighted'))

    recall.append(recall_score(y_test, y_predict_dt, average='weighted'))
    recall.append(recall_score(y_test, y_predict_nb, average='weighted'))
    recall.append(recall_score(y_test, y_predict_sgd, average='weighted'))
    recall.append(recall_score(y_test, y_predict_svc, average='weighted'))
    recall.append(recall_score(y_test, y_predict_rf, average='weighted'))

    f1.append(f1_score(y_test, y_predict_dt, average='weighted'))
    f1.append(f1_score(y_test, y_predict_nb, average='weighted'))
    f1.append(f1_score(y_test, y_predict_sgd, average='weighted'))
    f1.append(f1_score(y_test, y_predict_svc, average='weighted'))
    f1.append(f1_score(y_test, y_predict_rf, average='weighted'))

    return accuracy, precision, recall, f1


# train the rest of the data (give partial scores for the rest)
X_1200, X_0, y_1200, y_0 = get_train_test(tfidf_td[0:1191], total_data[0:1191], cat = 'materi', testsize = 1)
X_rest = tfidf_td[1191:]

# The next for loop generates a matrix that gives four scores of all six classifiers
# for cat in category:
#     X_train, X_test, y_train, y_test = get_train_test(tfidf_d, csv_data, cat = cat, testsize = 400)
#     print("")
#     print(get_pred_scores(X_train, X_test, y_train, y_test))
#     print("------------------------------------------------------")


# function R_score() gives the score of decision tree
def R_score(X_lr, y_lr):
    # model can be linear regression OR decision tree
    # model = LinearRegression()
    model = DecisionTreeRegressor(random_state = 0,max_features = 'sqrt', max_depth=7)



    model.fit(X_lr,y_lr)
    model.feature_importances_

    df_importance = pd.DataFrame(100 * model.feature_importances_,
                                 index = ['color','size','quality','comfort','price','material'], columns=['importance'])
    df_importance = df_importance.sort_values(by='importance', axis=0,
                                              ascending=True)
    df_importance.plot(kind='barh', color='r', legend=False)
    plt.xlabel('Importance (%)')
    plt.ylabel('Feature')
    plt.grid()
    plt.tight_layout()
    plt.savefig('Importance Level.png')

    y_predict = model.predict(X_lr)
    # compute R^2 and mean absolute error
    score = model.score(X_lr, y_lr)
    mae = mean_absolute_error(y_lr, y_predict)
    return score, mae

# test_Regression() only takes the dataset that we manually labeled as training set to fit regression model
def test_Regression():
    col_total = total_data['overall'][:1191].values
    par_socres_t = []
    for cat in category:
        par_socres_t.append(total_data[cat][:1191])
    par_socres_t = np.array(par_socres_t).T
    return R_score(par_socres_t, col_total)

# the next print gives the score of fitted regression model of the dataset that we manually labeled
# print(test_Regression())



col=total_data['overall'][1191:]
t_score = col.values

par_socres = []
for cat in category:
    X_1191, X_0, y_1191, y_0 = get_train_test(tfidf_td[0:1191], total_data[0:1191], cat=cat, testsize=1)
    X_rest = tfidf_td[1191:]
    dt, nb, sgd, svc, rf = prediction_with_tfidf(X_1191, y_1191, X_rest)
    par_socres.append(rf)
par_socres = np.array(par_socres).T


print("The R^2 score and the mean absolute error are:")
print(R_score(par_socres, t_score))

