import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#creating a function to perform logistic regression models quicker
def logistic_regression(X_train, y_train):
 
    #defining logistic regression function
    logit = LogisticRegression()
    #fitting the data into the model
    logit.fit(X_train, y_train)
    
    #creating a list comprehension for the column names
    names = [column for column in X_train.columns]
    #adding intercept to the end of the list
    names.append('intercept')
    #creating a dataframe from the regression coefficient values and intercept
    coeff = pd.DataFrame(np.append(logit.coef_, logit.intercept_)).T
    #renaming the column names with the list of names
    coeff.columns = names
    
    # 'logit.predict' predicts class labels for samples in the parenthesis
    y_pred = logit.predict(X_train)
    # 'predict_prob' predicts probability estimates
    # y_pred_proba = logit.predict_proba(X_train)
    
    #creates a confusion matrix to see how accurate the model is
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    
    #creating a copy of y_train
    label = y_train
    #renaming column in copy of y_train
    label = label.rename(columns={label.columns[0]:'label'})
    #creating labels out of unique values for 
    labels = sorted(label.label.unique())
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, target_names=labels, output_dict=True))
    return coeff, cm, class_report



## creating a function to perform decision trees quicker
def decision_tree(X_train, y_train, depth_number):

    #defining DecisionTreeClassifier and setting max depth number
    clf = DecisionTreeClassifier(max_depth= depth_number, random_state=123)
    #fitting the data to the model
    clf.fit(X_train, y_train)
    # 'logit.predict' predicts class labels for samples in the parenthesis
    y_pred = clf.predict(X_train)
    # 'predict_proba' predicts porbability estimates
    # y_pred_proba = clf.predict_proba(X_train)
    #creating a confusion matrix and storing it in a DataFrame
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    #creating a copy of y_train
    label = y_train
    #renaming column in copy of y_train
    label = label.rename(columns={label.columns[0]:'label'})
    #creating labels out of unique values for 
    labels = sorted(label.label.unique())
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, target_names=labels, output_dict=True))
    return cm, class_report




def random_forest(X_train, y_train, min_sample, maximum_depth):

    #defining Random Forest function and setting min_sample and max_depth
    rf = RandomForestClassifier(min_samples_leaf= min_sample , max_depth = maximum_depth, random_state = 123)
    #fitting the function
    rf.fit(X_train,y_train)
    #making a prediction
    y_pred = rf.predict(X_train)
    #creating a confusion matrix and storing it in a DataFrame
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    #creating a copy of y_train
    label = y_train
    #renaming column in copy of y_train
    label = label.rename(columns={label.columns[0]:'label'})
    #creating labels out of unique values for 
    labels = sorted(label.label.unique())
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, target_names=labels, output_dict=True))
    return cm, class_report




def kneighbors(X_train, y_train, n_neighbor):
    
    #defining the function and setting the neighbors
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    #fitting the function to the model
    knn.fit(X_train, y_train)
    #making a prediction
    y_pred = knn.predict(X_train)
    #creating a confusion matrix and storing it in a dataframe
    cm = pd.DataFrame(confusion_matrix(y_train, y_pred))
    #creating a copy of y_train
    label = y_train
    #renaming column in copy of y_train
    label = label.rename(columns={label.columns[0]:'label'})
    #creating labels out of unique values for 
    labels = sorted(label.label.unique())
    #creating a classification report and saving it as a DataFrame
    class_report = pd.DataFrame(classification_report(y_train, y_pred, target_names=labels, output_dict=True))
    return cm, class_report