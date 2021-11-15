from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

class classical_models():

    '''
    Fit and predict using SVC model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_SVC(feature,feature_test, y_train):
        clf_SVC = SVC()
        clf_SVC.fit(feature,y_train)
        predictions = clf_SVC.predict(feature_test)
        return predictions



    '''
    Fit and predict using K-Nearest Neighbors model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_KNN(feature,feature_test, y_train):
        n_neighbors_input = input("State the number of neighbors: ")
        clf_KNN = KNeighborsClassifier(n_neighbors=n_neighbors_input)
        clf_KNN.fit(feature,y_train)
        predictions = clf_KNN.predict(feature_test)
        return predictions




    '''
    Fit and predict using Decision Tree model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_DT(feature,feature_test, y_train):
        clf_dtc = DecisionTreeClassifier()
        clf_dtc.fit(feature,y_train)
        predictions = clf_dtc.predict(feature_test)
        return predictions



    '''
    Fit and predict using Randon Forest model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_RFC(feature,feature_test, y_train):
        clf_RFC = RandomForestClassifier()
        clf_RFC.fit(feature,y_train)
        predictions = clf_RFC.predict(feature_test)
        return predictions




    '''
    Fit and predict using Ada Boost Classifier model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_Ada(feature,feature_test, y_train):
        clf_ada = AdaBoostClassifier()
        clf_ada.fit(feature,y_train)
        predictions = clf_ada.predict(feature_test)
        return predictions



    '''
    Fit and predict using Extra Tree Classifier model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_Extra(feature,feature_test, y_train):
        clf_extra = ExtraTreesClassifier()
        clf_extra.fit(feature,y_train)
        predictions = clf_extra.predict(feature_test)
        return predictions




    '''
    Fit and predict using SDG Classifier model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_SGDC(feature,feature_test, y_train):
        clf_sgdc = SGDClassifier()
        clf_sgdc.fit(feature,y_train)
        predictions = clf_sgdc.predict(feature_test)
        return predictions



    '''
    Fit and predict using Logistic Regression model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_LogisticReg(feature,feature_test, y_train):
        clf_logistic_reg = LogisticRegression()
        clf_logistic_reg.fit(feature,y_train)
        predictions = clf_logistic_reg.predict(feature_test)
        return predictions




    '''
    Fit and predict using Multinomial Naive Bayes model.
    @params: X_train: train dense matrix
    @params: X_test: test dense matrix
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_MNB(X_train,X_test, y_train):
        clf_mnb = MultinomialNB()
        clf_mnb.fit(X_train,y_train)
        predictions = clf_mnb.predict(X_test)
        return predictions



    '''
    Fit and predict using XGBoost Classifier model.
    @params: feature: vectorized train features (sparse matrix)
    @params: feature_test: vectorized test features (sparse matrix)
    @params: y_train: train labels
    @return: predictions from the model run.
    '''
    def model_XGBoost(feature,feature_test, y_train):
        clf_sgdc = XGBClassifier()
        clf_sgdc.fit(feature,y_train)
        predictions = clf_sgdc.predict(feature_test)
        return predictions






