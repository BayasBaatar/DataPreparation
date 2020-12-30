import numpy as np
import pandas as pd
import csv
import json
import tflearn
from collections import Counter
import math
from imblearn.over_sampling import SMOTE
from numpy import isnan
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from parameters import Parameters
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class MyTest:

    def __init__(self):
        self.spark = SparkSession.builder.appName("MLOps").getOrCreate()

    # Reads JSON file and saves "config" as jsonData
    @staticmethod
    def readJSON():
        with open('data/model_data.json') as f:
            data = json.load(f)
            jsonData = data['config']
        return jsonData

    # Loads CSV file and applies configuration
    def loadCsvFile(self, file_name):
        spark_df = self.spark.read.format("csv") \
            .option("delimiter", ",") \
            .option("inferSchema", "true") \
            .option("header", True) \
            .load(Parameters.CSV_PATH + file_name)

        # Transform the Apache dataframe into Pandas dataframe
        df_data = spark_df.select('*').toPandas()

        # Reads data points from JSON
        json_data = MyTest.readJSON()
        man = json_data['list_of_datapoint']['mandatory']
        opt = json_data['list_of_datapoint']['optional']

        # Combines data points and saves them as preferred_datafields
        preferred_datafields = man + opt

        if preferred_datafields != '':
            df_data = df_data[preferred_datafields]
        return df_data

    # Converts into DATE_FORMAT
    @staticmethod
    def convertDateTime(df_data, column):
        df_data[column] = df_data[column].apply(
            lambda x: np.nan if (x == 'null' or x == '0' or x == 0 or x is None) else datetime.utcfromtimestamp(
                int(x)).strftime(Parameters.DATE_FORMAT))
        # print(df_data)
        return df_data

    # Data Cleaning
    @staticmethod
    def dataCleaning(df_data):
        # check how long the dataframe is
        # df = len(df_data)
        # print(df)

        # number of columns
        # df = df_data.columns
        # df = df.size
        # print(df)

        # number of Nan values in each column
        # for column in df_data.columns:
        #     df = df_data[column].isnull()
        #     print(column, " - ", df)

        # check if the values in the given column in unique
        # df_data['Employee_ID'].is_unique
        # print('Employee ID is unique')

        # Converts null and 0 into np.nan
        for column in df_data.columns:
            if column in ('Employee_ID', 'Education', 'Job_Satisfaction', 'Marital_Status'):
                pass
            elif column == 'Start_Date':
                df_data[column] = df_data[column].fillna(0)
                if column in 'Start_Date':
                    df_data = MyTest.convertDateTime(df_data, 'Start_Date')
            else:
                df_data[column] = df_data[column].apply(lambda x: np.nan if x == 'null' or x == '0' or x == 0 or
                                                                            x == None else x)
        # print(df_data)
        return df_data

    # Exports CSV
    @staticmethod
    def printCSV(df):
        df.to_csv(r'C:\Users\Bayas\Desktop\DataPreparation\export_dataframe.csv', index=False, header=True)
        return df

    # Checks missing values
    @staticmethod
    def checkMissingValues(df_clean):
        percent_missing = df_clean.isnull().sum() * 100 / len(df_clean)
        # print(percent_missing)
        return percent_missing

    # Get name of the columns with missing values
    @staticmethod
    def getMissingValueColumns(df_clean, required_percentage):
        columns = df_clean.columns
        percent = MyTest.checkMissingValues(df_clean)
        percent_missing_val = pd.DataFrame({'column_name': columns, 'percent_missing': percent}).sort_values(
            by='percent_missing', ascending=False)
        # print(percent_missing_val)

        # if "percent_missing" >50%, then noting as 'higher missing values'
        missing_val_columns = percent_missing_val.loc[
            percent_missing_val['percent_missing'] > required_percentage].index.tolist()
        # print(missing_val_columns)

        return missing_val_columns

    # Imputation
    def impute_once(self, df_clean):
        # data = self.loadCsvFile('HR-Employee-Dataset.csv')
        # dataframe = self.replaceNaN(data)
        # print(df_clean.columns)
        imputer = IterativeImputer()
        imputer.fit(df_clean)
        Xtrans = imputer.transform(df_clean)
        imputed_df = pd.DataFrame(Xtrans, columns=df_clean.columns)
        # print(imputed_df)
        # print('Missing after impute: %d' % sum(isnan(Xtrans).flatten()))
        # Utilities.exportCsvFile(imputed_df, file_name='missing_val_imputed_dataframe')

        return imputed_df

    def labelEncoding(self, df_data):
        # get categorical columns
        labelEncoding = preprocessing.LabelEncoder()
        catCols = df_data.select_dtypes("object").columns
        # print(catCols)
        for column in catCols:
            df_data[column] = labelEncoding.fit_transform(df_data[column])
        return df_data

    def dropColumn(self, df_clean):
        df_clean = df_clean.drop(
            ['Years_At_Company', 'Salary_Scale', 'Performance_Scale', 'Marital_Status'], axis=1)
        return df_clean

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

    def imbalanceAndModelTraining(self, df_clean):
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
            X = self.X_prepare(df_clean)
            y = self.y_prepare(y_sampling)
            # Splitting the dataset into the Training set and Test set
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

            #feature importance
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
            x_train_fea, x_test_fea, y_train_fea, y_test_fea = train_test_split(x_sampling, y_sampling, test_size=0.20, random_state=0)
            rfc = RandomForestClassifier(**rf_params)
            rfc.fit(x_train_fea, y_train_fea)
            importances = rfc.feature_importances_
            importances = pd.DataFrame({'feature': x_sampling, 'importance': 100 * np.round(importances, 3)})
            importances = importances.sort_values('importance', ascending=False).set_index('feature')

            print('-0-----------------------',importances)

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

    # Function to run all the functions
    def run(self):
        raw_data = self.loadCsvFile('HR-Employee-Dataset-Large.csv')
        df_clean = self.dataCleaning(raw_data)
        qualify_data = self.checkMissingValues(df_clean)
        qualified_data = self.getMissingValueColumns(df_clean, 50)
        # print(qualify_data)
        # print(qualified_data)
        # Drop missing columns
        df_clean = df_clean.drop(
            ['Years_At_Company', 'Salary_Scale', 'Performance_Scale', 'Standard_Hours'], axis=1)
        df_clean['Position'] = df_clean.groupby('Department').Position.transform(lambda x: x.fillna(x.mode()[0]))
        # print(df_clean)
        df_encode = self.labelEncoding(df_clean)
        df_clean = self.impute_once(df_encode)
        df_clean = self.imbalanceAndModelTraining(df_clean)
        # print(df_clean)
        # test.printCSV(df_clean)


if __name__ == '__main__':
    test = MyTest()
    test.run()
    # test.newColumn()
