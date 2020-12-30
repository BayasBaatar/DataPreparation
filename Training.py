import numpy as np
import pandas as pd
import tflearn
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class Training:
    @staticmethod
    def X_prepare(df_clean):
        # print(df_clean.columns)
        X_label = df_clean[['Position', 'Department', 'Gender', 'Education_Field']]
        # X_label_columns = X_label.columns
        X_number = df_clean[['Age', 'Education',
                             'Job_Satisfaction', 'Hourly_Rate', 'Monthly_Rate',
                             'Monthly_Income',
                             'Percent_Salary_Hike', 'Performance_Rating', 'Total_Working_Years',
                             'Work_Life_Balance', 'Start_Date'
                             ]]
        encoder = OneHotEncoder()
        X_label_onehot = encoder.fit_transform(X_label).toarray()
        scaler = StandardScaler()
        X_number_standard = scaler.fit_transform(X_number)
        # X_number_pd = pd.DataFrame(X_number_standard, columns=X_number.columns)
        X = np.concatenate((X_number_standard, X_label_onehot), axis=1)
        # print('X-->', pd.DataFrame(X))
        # print('X.shape: ', X.shape)
        return X

    @staticmethod
    def y_prepare(y_sampling):
        y = y_sampling[:, np.newaxis]
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y).toarray()
        # print(y.shape)
        return y

    @staticmethod
    def imbalanceAndModelTraining(df_clean):
        y_sampling = df_clean['is_active']
        x_sampling = df_clean.drop(columns=['is_active'])
        counter = Counter(y_sampling)
        # print(counter)

        if df_clean.isnull().sum().sum() == 0:
            # SMOTE
            oversample = SMOTE(random_state=42, k_neighbors=1)
            x_smote, y_smote = oversample.fit_sample(x_sampling, y_sampling)
            counter = Counter(y_smote)
            # print(counter)
            X = Training.X_prepare(df_clean)
            y = Training.y_prepare(y_sampling)
            # Splitting the dataset into the Training set and Test set
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

            # feature importance
            rf_params = {
                'n_jobs': -1,
                'n_estimators': 1000,
                #     'warm_start': True,
                'max_features': 0.3,
                'max_depth': 4,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': 0,
                'verbose': 0
            }
            x_train_fea, x_test_fea, y_train_fea, y_test_fea = train_test_split(x_sampling, y_sampling, test_size=0.20,
                                                                                random_state=0)
            rfc = RandomForestClassifier(**rf_params)
            rfc.fit(x_train_fea, y_train_fea)
            importances = rfc.feature_importances_
            importances = pd.DataFrame({'feature': x_sampling, 'importance': 100 * np.round(importances, 3)})
            importances = importances.sort_values('importance', ascending=False).set_index('feature')

            print('-0-----------------------', importances)

            # Model Training
            net = tflearn.input_data(shape=[None, X.shape[1]])
            net = tflearn.fully_connected(net, 6, activation='relu')
            net = tflearn.fully_connected(net, 6, activation='relu')
            net = tflearn.fully_connected(net, 6, activation='relu')
            net = tflearn.fully_connected(net, 2, activation='softmax')
            net = tflearn.regression(net)
            model = tflearn.DNN(net)
            model.fit(x_train, y_train, n_epoch=120, batch_size=32, show_metric=True)

            y_pred = model.predict(X)

            df_turnover = pd.DataFrame(
                {'emp_identifier': df_clean.Employee_ID, 'turnover_percent': 100 * y_pred[:, 0]})
            print(df_turnover)

            score_test = model.evaluate(x_test, y_test)
            print('X_test, y_test Accuracy: %0.4f%%' % (score_test[0] * 100))

        return df_clean
