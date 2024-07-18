import pandas as pd
import numpy as np
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

def generate_synthetic_data(num_records):
    fake = Faker()
    transaction_types = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
    payment_methods = ["Credit Card", "Debit Card", "Bank Transfer", "Mobile Payment"]
    statuses = ["Completed", "Pending", "Failed"]
    devices = ["Desktop", "Mobile", "Tablet"]
    browsers = ["Chrome", "Firefox", "Safari", "Edge"]
    geolocations = ["US", "UK", "CA", "DE", "FR"]
    data = []

    for _ in range(num_records):
        step = fake.random_int(min=1, max=30)
        transaction_type = random.choice(transaction_types)
        amount = round(random.uniform(1.0, 10000.0), 2)
        nameOrig = fake.uuid4()
        oldBal = round(random.uniform(0.0, 20000.0), 2)
        newBal = oldBal + amount if transaction_type == "CASH_IN" else oldBal - amount
        nameDest = fake.uuid4()
        oldBalDest = round(random.uniform(0.0, 20000.0), 2)
        newBalDest = oldBalDest - amount if transaction_type == "CASH_OUT" else oldBalDest + amount
        isFraud = random.choices([0, 1], weights=[0.98, 0.02])[0]
        isFlaggedFraud = 1 if isFraud == 1 and amount > 5000 else 0
        transaction_id = fake.uuid4()
        timestamp = fake.date_time_this_year()
        payment_method = random.choice(payment_methods)
        status = random.choice(statuses)
        user_id = fake.uuid4()
        login_time = fake.date_time_this_year()
        device = random.choice(devices)
        browser = random.choice(browsers)
        geolocation = random.choice(geolocations)
        activity = fake.sentence()

        data.append([
            step, transaction_type, amount, nameOrig, oldBal, newBal, nameDest,
            oldBalDest, newBalDest, isFraud, isFlaggedFraud, transaction_id, timestamp,
            payment_method, status, user_id, login_time, device, browser, geolocation, activity
        ])
    
    columns = [
        "Step", "Type", "Amount", "nameOrig", "oldBal", "newBal", "nameDest",
        "oldBalDest", "newBalDest", "isFraud", "isFlaggedFraud", "TransactionId",
        "Timestamp", "Payment Method", "Status", "UserId", "LoginTime", "DeviceType",
        "Browser", "Geolocation", "Activity"
    ]
    return pd.DataFrame(data, columns=columns)

def preprocess_data(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['LoginTime'] = pd.to_datetime(df['LoginTime'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['IsBusinessHours'] = df['Hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
    df['NormalizedAmount'] = df['Amount'] / (df['oldBal'] + 1)
    df['AmountPercentage'] = (df['Amount'] / (df['oldBal'] + 1)) * 100
    df['OriginBalanceChange'] = df['newBal'] - df['oldBal']
    df['DestBalanceChange'] = df['newBalDest'] - df['oldBalDest']
    top_devices = df['DeviceType'].value_counts().index[:3]
    df['IsCommonDevice'] = df['DeviceType'].apply(lambda x: 1 if x in top_devices else 0)
    top_browsers = df['Browser'].value_counts().index[:3]
    df['IsCommonBrowser'] = df['Browser'].apply(lambda x: 1 if x in top_browsers else 0)
    df = df.drop(columns=['Timestamp', 'LoginTime', 'TransactionId', 'UserId', 'nameOrig', 'nameDest', 'Activity'])
    df['Type'] = df['Type'].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
    df['Payment Method'] = df['Payment Method'].astype('category').cat.codes
    df['DeviceType'] = df['DeviceType'].astype('category').cat.codes
    df['Browser'] = df['Browser'].astype('category').cat.codes
    df['Geolocation'] = df['Geolocation'].astype('category').cat.codes
    df['Status'] = df['Status'].astype('category').cat.codes
    df['isFraud'] = df['isFraud'].map({0: 0, 1: 1})
    return df.dropna()

def split_and_balance_data(df):
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, X_test, y_train_balanced, y_test

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return rf_classifier, accuracy, conf_matrix, class_report

def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_tuned_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return accuracy, conf_matrix, class_report, roc_auc, y_pred_proba

def plot_roc_curve(y_test, y_pred_proba, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def save_model(model, filename):
    joblib.dump(model, filename)

def feature_importances(model, X):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    return feature_importance_df.sort_values(by='Importance', ascending=False)

# Main script execution
num_records = 1000
df = generate_synthetic_data(num_records)
df = preprocess_data(df)
df.to_csv('engineered_fraud_detection_data.csv', index=False)

X_train_balanced, X_test, y_train_balanced, y_test = split_and_balance_data(df)
rf_classifier, accuracy, conf_matrix, class_report = train_and_evaluate_model(X_train_balanced, y_train_balanced, X_test, y_test)
print(f'Initial Accuracy: {accuracy:.2f}')
print('Initial Confusion Matrix:', conf_matrix)
print('Initial Classification Report:', class_report)

best_rf_classifier = tune_model(X_train_balanced, y_train_balanced)
accuracy, conf_matrix, class_report, roc_auc, y_pred_proba = evaluate_tuned_model(best_rf_classifier, X_test, y_test)
print(f'Tuned Accuracy: {accuracy:.2f}')
print('Tuned Confusion Matrix:', conf_matrix)
print('Tuned Classification Report:', class_report)
print(f'AUC-ROC: {roc_auc:.2f}')

plot_roc_curve(y_test, y_pred_proba, roc_auc)
save_model(best_rf_classifier, 'best_rf_classifier.pkl')
feature_importance_df = feature_importances(best_rf_classifier, X_test)
print(feature_importance_df)
