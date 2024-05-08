import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
import csv

def load_ratingdata_ML():

    df = np.zeros(shape=(6040, 3952))
    user_genre_pref = []
    user_ratings = {}

    user = set()

    with open("ml-1m/merge_rating.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating,  genres, gender = line.split(":")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)

            # user-item interaction matrix formation
            if user_id <= 6040 and movie_id <= 3952:
                df[user_id - 1, movie_id - 1] = rating
                user.add(user_id)

            # data density measure
            count = 0
            if user_id <= 6040 and movie_id <= 3952 and df[user_id - 1, movie_id - 1] != 0:
                count +=1
            density = (count/df.size)*100
            # end data density measure

            user_genre_pref.append((user_id, movie_id, gender, genres.strip().split('|')))
            if user_id not in user_ratings:
                user_ratings[user_id] = []  # Initialize list for user_id if it doesn't exist
            user_ratings[user_id].append((movie_id, rating, genres.split('|'), gender))

    print(density)
    num_unique_users = len(user)
    return df, user_genre_pref, user_ratings, num_unique_users

def load_genderdata_ML():
    gender_vec = []

    # to count male & female in data
    m = 0
    fm = 0

    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines()[:6040]:
            user_id, gender, _, _, _ = line.split("::")
            if gender == "M":
                gender_vec.append(0)
                m +=1
            else:
                gender_vec.append(1)
                fm +=1

    print('male: ', m, ' female: ', fm)
    return np.asarray(gender_vec)

def load_ratingdata_yahoo():

    movies = set()  # Using set to automatically deduplicate
    users = set()   # Using set to automatically deduplicate
    ratings = []
    user_genre_pref = []
    user_ratings = {}

    with open('yahoo_movie_dataset/yahoo_mergerating.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            userid,movieid,Rating,Genres,gender = row
            user_id = int(userid)
            movie_id = int(movieid)
            rating = float(Rating)
            movies.add(movie_id)
            users.add(user_id)
            ratings.append((user_id, movie_id, float(rating)))
            user_genre_pref.append((user_id, movie_id, gender, Genres.strip().split('|')))

            if user_id not in user_ratings:
                user_ratings[user_id] = []  # Initialize list for user_id if it doesn't exist
            user_ratings[user_id].append((movie_id, rating, Genres.split('|'), gender))


    num_unique_movies = len(movies)
    num_unique_users = len(users)
    print(num_unique_users, num_unique_movies)

    df = np.zeros(shape=(num_unique_users, num_unique_movies))

    for user_id, movie_id, rating in ratings:
         df[user_id-1, movie_id-1] = rating

    count = 0
    if user_id in range(num_unique_users) and movie_id in range(num_unique_movies) and df[user_id - 1, movie_id - 1] != 0:
        count += 1

    density = (count / df.size) * 100

    print(density)
    return df, user_genre_pref, user_ratings, num_unique_users

def load_genderdata_yahoo():
    gender = []
    m = []
    m_count = 0
    f_count = 0
    with open('yahoo_movie_dataset/update_users.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            userid,gender_val, uid = row
            if gender_val == 'm':
                gender.append(1)
                m.append(1)
                m_count +=1
            else:
                gender.append(0)
                f_count +=1
    print('male count: ', m_count, " & female count: ", f_count)
    print('m lenght', len(m))
    return np.asarray(gender)


def get_genre_preferences(user_genre_pref, user_ratings, num_unique_users):

    male_pref_genres = {'Action', 'Horror', 'War', 'Crime', 'Adventure', 'Comedy'}
    female_pref_genres = {'Romance', 'Drama', "Children's", 'Animation'}

    user_info=[]

    male_preferences = {user_id: 0 for user_id in range(num_unique_users)}  # Initialize with default value 0 for all user IDs
    female_preferences = {user_id: 0 for user_id in range(num_unique_users)}
    user_total_genres_count = {user_id: 0 for user_id in range(num_unique_users)}
    gender_data = {user_id: 0 for user_id in range(num_unique_users)}
    user_avg_ratings = {user_id: 0 for user_id in range(num_unique_users)}

    for user_id, movie_id, gender, genres in user_genre_pref:
        gender_data[user_id-1] = gender
        user_total_genres_count[user_id-1] += len(genres)
        female_preferences[user_id - 1] += sum(genre in female_pref_genres for genre in genres)
        male_preferences[user_id - 1] += sum(genre in male_pref_genres for genre in genres)

    for user_id in range(6040):
        if gender_data[user_id] =='F\n' and male_preferences[user_id] > female_preferences[user_id]:
            user_info.append((user_id+1, gender_data[user_id], male_preferences[user_id], female_preferences[user_id],
                              user_total_genres_count[user_id]))
        elif gender_data[user_id] =='M\n' and male_preferences[user_id] < female_preferences[user_id]:
            user_info.append((user_id+1, gender_data[user_id], male_preferences[user_id], female_preferences[user_id],
                              user_total_genres_count[user_id]))

    return np.asarray([[v] for k, v in male_preferences.items()]), \
        np.asarray([[v] for k, v in female_preferences.items()]), \
        user_info, np.asarray([[v] for k, v in user_avg_ratings.items()]), \
        np.asarray([[v] for k, v in user_total_genres_count.items()])

def normalize(X):
    from sklearn import preprocessing
    X = preprocessing.normalize(X)
    return X

# Evaluate model and compute AUC
def compute_auc(X, T, model):
    probs = model.predict_proba(X)
    preds = probs[:, 1]
    auc = roc_auc_score(T, preds)
    return auc


def matthews_corrcoef(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        return 0

    return numerator / denominator

# Load rating data and user genre preferences ML
X, user_genre_pref, user_ratings, num_unique_users = load_ratingdata_ML()
y = load_genderdata_ML()

# Load rating data and user genre preferences yahoo
#X, user_genre_pref, user_ratings, num_unique_users = load_ratingdata_yahoo()
#y = load_genderdata_yahoo()

male_prefer, female_prefer, info, ave_rate, genre_count = get_genre_preferences(user_genre_pref, user_ratings, num_unique_users)

# RQ2.1
print(info)
print(len(info))
# end RQ2.1

X_with_genre = np.concatenate((male_prefer, female_prefer), axis=1)
print("Shape of X:", X.shape)

# Normalize data
X_with_genre = normalize(X_with_genre)
X= normalize(X)

X_with_genre_pref = np.concatenate((X, X_with_genre), axis=1)
print('shape of X_with_genre_pref after cat: ', X_with_genre_pref.shape)


# Split data into train and test sets
#RQ2
X_train, X_test, y_train, y_test = train_test_split(X_with_genre_pref, y, test_size=0.2, random_state=42)
X_Rtrain, X_Rtest, y_Rtrain, y_Rtest = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- grid search
# Define parameter grid
param_grid = {
    'penalty': ['l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

#logistic_regression = LogisticRegression(max_iter=1000)
# Create GridSearchCV object
#grid_search = GridSearchCV(logistic_regression, param_grid, cv=10, scoring='roc_auc')
#grid_search.fit(X_with_genre_pref, y)

# Access the best estimator and best parameters
#best_model = grid_search.best_estimator_
#best_params = grid_search.best_params_

#print("Best model:", best_model)
#print("Best parameters:", best_params)

# ----- end grid search

# RQ2: classifiers

def drawConfusionMatrix(test, pred):

    cf_matrix = confusion_matrix(test, pred)
    sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()


def classifierLR(X_train, y_train, X_test, y_test):
    model = LogisticRegression(penalty='l2', C=1.0, random_state=0, max_iter=1000)
    model.fit(X_train, y_train)

    print("--- Logistic Regression ---")

    auc = compute_auc(X_test, y_test, model)
    print("AUC :", auc)

    # plot confusion matrix
    y_pred = model.predict(X_test)
    drawConfusionMatrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print('MCC', matthews_corrcoef(y_test, y_pred))

def classifieradaboost(X_train, y_train, X_test, y_test):

    model = AdaBoostClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    print("--- ADABoost ---")

    auc = compute_auc(X_test, y_test, model)
    print("AUC:", auc)

    # plot confusion matrix
    y_pred = model.predict(X_test)
    drawConfusionMatrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print('MCC: ', matthews_corrcoef(y_test, y_pred))

def classifierXG(X_train, y_train, X_test, y_test):

    model = xgb.XGBClassifier(objective='multi:softmax', num_class=2, random_state=42)
    model.fit(X_train, y_train)

    print("--- XGBoost ---")

    AUC = compute_auc(X_test, y_test, model)
    print("AU:", AUC)

    # plot confusion matrix
    y_pred = model.predict(X_test)
    drawConfusionMatrix(y_test, y_pred)

    accuracy_XG = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy_XG)
    print('MCC: ', matthews_corrcoef(y_test, y_pred))

def classifierSVM(X_train, y_train, X_test, y_test):

    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)

    print("--- SVM ---")

    y_pred = model.predict(X_test)

    # Getting predicted probabilities for calculating AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print("AUC:", auc)

    drawConfusionMatrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print('MCC SVM', matthews_corrcoef(y_test, y_pred))

#print("GS with Rating")
#classifierLR(X_train, y_train, X_test, y_test)
#classifieradaboost(X_train, y_train, X_test, y_test)
#classifierXG(X_train, y_train, X_test, y_test)
#classifierSVM(X_train, y_train, X_test, y_test)

#print("Rating only")
#classifierLR(X_Rtrain, y_Rtrain, X_Rtest, y_Rtest)
#classifieradaboost(X_Rtrain, y_Rtrain, X_Rtest, y_Rtest)
#classifierXG(X_Rtrain, y_Rtrain, X_Rtest, y_Rtest)
#classifierSVM(X_Rtrain, y_Rtrain, X_Rtest, y_Rtest)

def crossvalidation(X_with_genre_pref, y):

    # Initialize StratifiedKFold with 10 folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    auc_scores = []
    accuracy_scores = []
    precision = []
    recall = []
    f1 = []
    # Perform 10-fold cross-validation
    for train_index, test_index in skf.split(X_with_genre_pref, y):

        X_train, X_test = X_with_genre_pref[train_index], X_with_genre_pref[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit logistic regression model
        model = LogisticRegression(penalty='l2', C=1.0, random_state=0, max_iter=1000)
        model.fit(X_train, y_train)

        # fit SVM
        #model = SVC(kernel='linear', probability=True, random_state=42)
        #model.fit(X_train, y_train)
        #y_proba = model.predict_proba(X_test)[:, 1]

        # Compute AUC on test set
        auc = compute_auc(X_test, y_test, model)
        #auc = roc_auc_score(y_test, y_proba) # for svm
        auc_scores.append(auc)

        # Compute accuracy on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1= f1_score(y_test, y_pred)

    # Compute average AUC and accuracy
    average_auc = np.mean(auc_scores)
    average_accuracy = np.mean(accuracy_scores)
    precision_check = np.mean(precision)
    recall_check = np.mean(recall)
    f1_check = np.mean(f1)

    print("Average AUC: {:.3f}".format(average_auc), " Average Accuracy: {:.3f}".format(average_accuracy))
    print('Ave Precision: {:.3f}'.format(precision_check), ' ave recall: {:.3f}'.format(recall_check), ' F1: {:.3f}'.format(f1_check))

print('LR')
print('crossvalidation(X_with_genre_pref, y)')
crossvalidation(X_with_genre_pref, y)
print('crossvalidation(X, y)')
crossvalidation(X, y)

#print('SVM')
#print('crossvalidation(X_with_genre_pref, y)')
#crossvalidation(X_with_genre_pref, y)
#print('crossvalidation(X, y)')
#crossvalidation(X, y)

