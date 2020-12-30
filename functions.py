import json
import pandas as pd
from parameters import Parameters


class Functions:

    # Reads JSON file and saves "config" as jsonData
    @staticmethod
    def readJSON():
        with open('data/model_data.json') as f:
            data = json.load(f)
            jsonData = data['config']
        return jsonData

    # Loads CSV file and applies configuration
    @staticmethod
    def loadCsvFile(spark, file_name):
        spark_df = spark.read.format("csv") \
            .option("delimiter", ",") \
            .option("inferSchema", "true") \
            .option("header", True) \
            .load(Parameters.CSV_PATH + file_name)

        # Transform the Apache dataframe into Pandas dataframe
        df_data = spark_df.select('*').toPandas()

        # Reads data points from JSON
        json_data = Functions.readJSON()
        man = json_data['list_of_datapoint']['mandatory']
        opt = json_data['list_of_datapoint']['optional']

        # Combines data points and saves them as preferred_datafields
        preferred_datafields = man + opt

        if preferred_datafields != '':
            df_data = df_data[preferred_datafields]
        return df_data

    # Checks missing values
    @staticmethod
    def checkMissingValues(df_clean):
        percent_missing = df_clean.isnull().sum() * 100 / len(df_clean)
        return percent_missing

    # Get name of the columns with missing values
    @staticmethod
    def getMissingValueColumns(df_clean, required_percentage):
        columns = df_clean.columns
        percent = Functions.checkMissingValues(df_clean)
        percent_missing_val = pd.DataFrame({'column_name': columns, 'percent_missing': percent}).sort_values(
            by='percent_missing', ascending=True)
        # print(percent_missing_val)

        # if "percent_missing" >50%, then noting as 'higher missing values'
        missing_val_columns = percent_missing_val.loc[
            percent_missing_val['percent_missing'] > required_percentage].index.tolist()
        # print("Columns with more than 50% of the data missing: ", missing_val_columns)

        return missing_val_columns

    @staticmethod
    def missingValueImputation(df_clean):
        df_clean['Position'] = df_clean.groupby('Department').Position.transform(lambda x: x.fillna(x.mode()[0]))
        return df_clean

    @staticmethod
    def dropColumns(df_clean, column):
        df_clean = df_clean.drop(column, axis=1)
        return df_clean
