import numpy as np
import pandas as pd
from datetime import datetime
from parameters import Parameters
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing


class DataCleansing:
    # Converts into DATE_FORMAT
    @staticmethod
    def convertDateTime(df_data, column):
        df_data[column] = df_data[column].apply(
            lambda x: np.nan if (x == 'null' or x == '0' or x == 0 or x is None or x == np.nan) else datetime.utcfromtimestamp(
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
            elif column == 'Years_At_Company':
                df_data[column] = df_data[column].fillna(0)
                if column in 'Years_At_Company':
                    df_data = DataCleansing.convertDateTime(df_data, 'Years_At_Company')
            elif column == 'Start_Date':
                df_data[column] = df_data[column].fillna(0)
                if column in 'Start_Date':
                    df_data = DataCleansing.convertDateTime(df_data, 'Start_Date')
            else:
                df_data[column] = df_data[column].apply(lambda x: np.nan if x == 'null' or x == '0' or x == 0 or
                                                                            x == None else x)
        return df_data

    # Imputation
    @staticmethod
    def imputeOnce(df_clean):
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

    @staticmethod
    def labelEncoding(df_data):
        # get categorical columns
        labelEncoding = preprocessing.LabelEncoder()
        catCols = df_data.select_dtypes("object").columns
        # print(catCols)
        for column in catCols:
            df_data[column] = labelEncoding.fit_transform(df_data[column])
        return df_data
