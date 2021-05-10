import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,explained_variance_score
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

def main():
    st.title("Your Binary Model Web App")
    st.sidebar.title("Binary Model Web App")
    st.markdown("Determine the accurate Model's algorithm ")
    st.sidebar.markdown("Model accuracy is affected by Outlier")


    st.sidebar.subheader("Select The supervised learning mode")
    SuperVised = st.sidebar.selectbox("", ('Classification','Regression'), key='SuperVised')


    st.sidebar.subheader("Select splitting mode")
    CrosVal = st.sidebar.checkbox("Cross Validation", True)


    @st.cache(persist=True)
    def load_data():



        df = pd.read_csv('C:\\Users\\Dell\\Desktop\\Pr1\\data\\cardio1.csv')
        return df

    @st.cache(persist=True)
    def split(df):

            my_features = ['age_year', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco','active']
            Input = df[my_features]

            labels = df['cardio']



            reg_feature = ['age_year']
            inputReg = df[reg_feature]

            labelReg = df['ap_hi']



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


            # ---------------------------------------------------------------------#
            if SuperVised == ['Regression']:
                kfold = KFold(n_splits=10, random_state=40, shuffle=True)
                for train_index, test_index in kfold.split(inputReg):
                    x_train, x_test, y_train, y_test = inputReg.iloc[train_index], inputReg.iloc[test_index], \
                                                       labelReg.iloc[train_index], labelReg.iloc[test_index]
                    return x_train, x_test, y_train, y_test
            else:
                # CrosVal = st.sidebar.checkbox("Cross Validation") ==False
                if CrosVal == 'False':
                        x_train, x_test, y_train, y_test = train_test_split(Input, labels, test_size=0.3, random_state=0)
                        return x_train, x_test, y_train, y_test
                else:
                    kfold = KFold(n_splits=10, random_state=40, shuffle=True)
                    for train_index, test_index in kfold.split(Input):
                        x_train, x_test, y_train, y_test = Input.iloc[train_index], Input.iloc[test_index],\
                                                               labels.iloc[train_index], labels.iloc[test_index]
                        return x_train, x_test, y_train, y_test


    #                st.sidebar.subheader("Select splitting mode")

    # ------------------------------------------------------------------


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
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

# klearn.metrics.explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
    df = load_data()
    class_names = ['no-stroke', 'stroke']

    x_train, x_test, y_train, y_test = split(df)


    over = SMOTE()
    x_train1, y_train1 = over.fit_resample(x_train, y_train)

    if SuperVised =='Classification':
        st.sidebar.subheader("Select Classifier")
        classifier = st.sidebar.selectbox("Choose one Classifier",
                                          ("Random Forest","Logistic Regression", "Decision Tree Classifier","KNNeighbors Classifier"))
    if SuperVised =='Regression':
        st.sidebar.subheader("Select Model")
        classifier = st.sidebar.selectbox("Choose one Classifier",
                                          ("Decision Tree Regression","Random Forest Regression","Linear Regression"))


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Classifier's Parameters")
        C = st.sidebar.number_input("Regularization Value", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        Fimportance = st.sidebar.radio("Select importance", ('False','True'), key='Fimportance')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve','Precision-Recall Curve'))

        if  st.sidebar.button("Classify", key='classify'):
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
        smote     = st.sidebar.radio("SMOTE", ('True', 'False'), key='smote')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
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
                                         ('Confusion Matrix', 'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Decision Tree Classifier Results")
            model = DecisionTreeClassifier() #   (C=C, max_iter=max_iter)
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
                                         ('Confusion Matrix', 'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
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

# -Regression---------------------------------------------------------------------------------Linear-Regression

    if classifier == 'Linear Regression':
        if  st.sidebar.button("Regress", key='classify'):
            st.subheader("Linear Regression Results")
            model = LinearRegression()
            model.fit(x_train1, y_train1)
            accuracy1 = model.score(x_train1, y_train1)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy on train ", accuracy1.round(4))
            st.write("Accuracy on test", accuracy.round(4))
#            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
#            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
            st.write('The explained variance of the Linear Regression model of Cardio data is:',
            explained_variance_score(y_test, y_pred))

           # ------------------------------------ ----------------------------------------------------------Random-Forest-Regression-----
    if classifier == 'Random Forest Regression':
        st.sidebar.subheader("Regressor's Parameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 10, 100, step=1,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='max_depth')
       # bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        smote     = st.sidebar.radio("SMOTE", ('True', 'False'), key='smote')

        if st.sidebar.button("Regress", key='classify'):
            st.subheader("")
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth ,n_jobs=-1) # , bootstrap=bootstrap)
            if smote == 'True':
               model.fit(x_train1, y_train1)
            else:
                 model.fit(x_train, y_train)
            accuracy1 = model.score(x_train1, y_train1)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy on train ", accuracy1.round(4))
            st.write("Accuracy on test", accuracy.round(4))
#            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
#            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
            st.write('The explained variance of the Random Forest Regression model of Cardio data is:',
            explained_variance_score(y_test, y_pred))

           # ------------------------------------ ----------------------------------------------------------Decision-Tree-Regression-----
    if classifier == 'Decision Tree Regression':
        st.sidebar.subheader("Regressor's Parameters")
        DTR_max_dep = st.sidebar.number_input("max_depth", 1, 500,  step=1, key='DTR_max_dep')
        DTR_min_samples_split = st.sidebar.number_input("min_samples_split", 1, 500, key='min_samples_split')

        if st.sidebar.button("Regress", key='classify'):
            st.subheader("Decision Tree Classifier Results")
            model = DecisionTreeRegressor(max_depth=DTR_max_dep, min_samples_split=DTR_min_samples_split) #   (C=C, max_iter=max_iter)
            model.fit(x_train1, y_train1)
            accuracy1 = model.score(x_train1, y_train1)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy on train ", accuracy1.round(4))
            st.write("Accuracy on test", accuracy.round(4))
#            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(4))
#            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(4))
            st.write('The explained variance of the Decision Tree Regression model of Cardio data is:',
            explained_variance_score(y_test, y_pred))


# -----------------------------------------------------------------------end of Regression


    if st.sidebar.checkbox("Show raw data", False):
       st.subheader("Cardio Data Set (Classification)")
       st.write(df)


    st.sidebar.subheader("SMOTE: Synthetic Minority Oversampling Technique")
    st.sidebar.write("Date Before SMOTE ", Counter(y_train))
    st.sidebar.write("Date After SMOTE ", Counter(y_train1))




if __name__ == '__main__':
    main()

