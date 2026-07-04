import streamlit as st
import pandas as pd

from utils import issues as issue_module
issue_module.issues = []

st.title("ML Debugger")
file = st.file_uploader(
    "Upload your dataset (CSV only, max 25MB): ",
    type=["csv"],
    max_upload_size=25,
)

#data preview
if file:
    df = pd.read_csv(file)

    st.subheader("Data Preivew")
    st.write(df.head())

#basic info function
from utils.analysis import basic_info
if file :
    info = basic_info(df)
    st.subheader("Basic Info")
    st.write(f"Rows: {info['rows']}")
    st.write(f"Columns: {info['columns']}")
    st.write("Column Names:", info["Column_list"])

from utils.analysis import missing_percent
if file:
    missing_per_val = missing_percent(df)
    st.subheader("Percentage of missing values per column")
    st.write(missing_per_val)

from utils.issues import missing_check
if file:
    max_imb_value = missing_check(df)
    if max_imb_value:
        st.error(max_imb_value)


from utils.analysis import class_coverage
if file:
    value = class_coverage(df)
    st.subheader("Class Imbalance Analysis")
    st.write(value)

from utils.issues import max_class
if file:
    max_imb_value = max_class(df)
    if max_imb_value:
        st.error(max_imb_value)

from utils.analysis import correlation_matrix
if file:
    correlation_matrix_val = correlation_matrix(df)
    st.subheader("Correlation Matrix")
    st.write(correlation_matrix_val)


from utils.issues import high_corr
if file:
    high_corr_prob = high_corr(df)
    if high_corr_prob:
        st.write("The following are the columns with high correlation that you need to get rid off: ")
        for i, j, k in high_corr_prob:
            st.warning(f"⚠ Features {i} and {j} are highly correlated {k}")



#the follwoing is going to be the baseline of the model 

#first we will specify x and y in the dataset 
if file:
    df = df.select_dtypes(include = 'number')
    df = df.dropna()
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
#stopping with ratio if regression model
    total = len(y)
    unique = y.nunique()

    ratio = unique / total
    if ratio > 0.8:
        st.error("Regression dataset detected. Only classification supported.")
        st.stop()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# Logistic Regression
from sklearn.linear_model import LogisticRegression

if file:
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    #Predicting and testing accuracy
    y_pred = lr.predict(X_test)

    st.subheader("Evaluation metrics and Result after training with logistic regression")
    #Computing the accuracy and model evaluation
    from sklearn.metrics import accuracy_score, confusion_matrix
    from utils.issues import high_corr_flag
    acc = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", acc)
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    if cm.shape != (2, 2):
        st.error("Only binary classification supported yet")
        st.stop()
    flag_high = high_corr_flag
    tn, fp, fn, tp = cm.ravel()

    recall_0 = tn / (tn + fp)
    recall_1 = tp / (tp + fn)

    if recall_0 == 0:
        st.error("Model completely ignores class 0")
    if recall_0 <= 0.35 and recall_0 > 0 and flag_high:
        st.warning("Model performs poorly on class 0 likely due to high imbalance of class")
    elif recall_0 <= 0.35 and recall_0 > 0:
        st.warning("Model performs poorly on class 0")

    if recall_1 == 0:
        st.error("Model completely ignores class 1")
    if recall_1 <= 0.35 and recall_1 > 0 and flag_high:
        st.warning("Model performs poorly on class 1 likely due to high imbalance of class")
    elif recall_1 <= 0.35 and recall_1 > 0 :
        st.warning("Model performs poorly on class 1")


from utils.issues import issues_list
if file:
    issues = issues_list()
    if issues:
        st.subheader("Recommended Actions to Improve Model Performance")
        for i, (msg, level) in enumerate(issues, 1):
            st.info(f"{i}. {msg} ({level})")