import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

def main():
    st.title("Your Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Any risk of Heart Attack??")
    st.sidebar.markdown("Check the Heart's Health")

    st.sidebar.subheader("Select splitting mode")
    CrosVal = st.sidebar.radio("True: Cross Validation, False: Splitting", ('True', 'False'), key='CrosVal')



    @st.cache(persist=True)
    def load_data():
        df = pd.read_csv('data/cardio1.csv')
        return df

    @st.cache(persist=True)
    def split(df):
            labels = df['cardio']

            my_features = ['age_year', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco',
                           'active']

            # convert features into numeric
            df['age_year'] = pd.to_numeric(df['age_year'], errors='coerce')
            df['height'] = pd.to_numeric(df['height'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['ap_hi'] = pd.to_numeric(df['ap_hi'], errors='coerce')
            df['ap_lo'] = pd.to_numeric(df['ap_lo'], errors='coerce')
            df['cholesterol'] = pd.to_numeric(df['cholesterol'], errors='coerce')
            df['gluc'] = pd.to_numeric(df['gluc'], errors='coerce')
            df['smoke'] = pd.to_numeric(df['smoke'], errors='coerce')
            df['alco'] = pd.to_numeric(df['alco'], errors='coerce')
            df['active'] = pd.to_numeric(df['active'], errors='coerce')

            # fill nan values with 1
            df = df.replace(np.nan, 1, regex=True)
            df = df.fillna(1)

            Input = df[my_features]
            if  CrosVal == ['False']:
                x_train, x_test, y_train, y_test = train_test_split(Input, labels, test_size=0.3, random_state=0)
                return x_train, x_test, y_train, y_test
            else:
                 kfold = KFold(n_splits=10, random_state=40, shuffle=True)
                 for train_index, test_index in kfold.split(Input):
                     x_train, x_test, y_train, y_test = Input.iloc[train_index], Input.iloc[test_index],\
                                                        labels.iloc[train_index], labels.iloc[test_index]
                     return x_train, x_test, y_train, y_test







    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test, alpha=0.8)
            st.pyplot()



    df = load_data()
    class_names = ['no-stroke', 'stroke']

    x_train, x_test, y_train, y_test = split(df)

  #  st.sidebar.subheader("select Smote")
  #   smote = st.sidebar.selectbox("Smote", ("SMOTE", "NOSMOTE"))



    over = SMOTE()
    x_train1, y_train1 = over.fit_resample(x_train, y_train)




    st.sidebar.subheader("Select Classifier")
    classifier = st.sidebar.selectbox("Choose one Classifier",
                                      ("Random Forest","Logistic Regression", "Decision Tree Classifier","KNNeighbors Classifier"))


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Classifier's Parameters")
        C = st.sidebar.number_input("Regularization Value", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        Fimportance = st.sidebar.radio("Select importance", ('False','True'), key='Fimportance')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve'))

        if  st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train1, y_train1)
            accuracy1 = model.score(x_train1, y_train1)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy on train ", accuracy1.round(4))
            st.write("Accuracy on test", accuracy.round(4))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
            plot_metrics(metrics)
            if Fimportance == 'True':
               importance = model.coef_[0]
               for i, v in enumerate(importance):
                   st.write('Feature: %0d, Score: %.5f' % (i, v))
                   pyplot.bar([x for x in range(len(importance))], importance)
               st.pyplot()

    if classifier == 'Random Forest':
        st.sidebar.subheader("Classifier's Parameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 1, 5000, step=1,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='max_depth')
       # bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        smote     = st.sidebar.radio("smote??", ('True', 'False'), key='smote')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth ,n_jobs=-1) # , bootstrap=bootstrap)
            if smote == 'True':
               model.fit(x_train1, y_train1)
            else:
                 model.fit(x_train, y_train)
            accuracy1 = model.score(x_train1, y_train1)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy on train ", accuracy1.round(4))
            st.write("Accuracy on test", accuracy.round(4))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
            plot_metrics(metrics)


    if classifier == 'Decision Tree Classifier':
        st.sidebar.subheader("Classifier's Parameters")
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Decision Tree Classifier Results")
            model = LogisticRegression() #   (C=C, max_iter=max_iter)
            model.fit(x_train1, y_train1)
            accuracy1 = model.score(x_train1, y_train1)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy on train ", accuracy1.round(4))
            st.write("Accuracy on test", accuracy.round(4))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
            plot_metrics(metrics)

    if classifier == 'KNNeighbors Classifier':
        st.sidebar.subheader("Classifier's Parameters")
        n_neighbors = st.sidebar.number_input("The number of neighbours in the classifier", 1, 10, step=1,key='n_neighbors')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("KNNeighbors Classifier Results")
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_train1, y_train1)
            accuracy1 = model.score(x_train1, y_train1)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy on train ", accuracy1.round(4))
            st.write("Accuracy on test", accuracy.round(4))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
       st.subheader("Cardio Data Set (Classification)")
       st.write(df)


    st.sidebar.subheader("Data on SMOTE")
    st.sidebar.write("Date Before SMOTE ", Counter(y_train))
    st.sidebar.write("Date Before SMOTE ", Counter(y_train1))




if __name__ == '__main__':
    main()

