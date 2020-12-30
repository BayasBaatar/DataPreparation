from pyspark.sql import SparkSession
from functions import Functions
from DataCleansing import DataCleansing
from Training import Training

func = Functions()


class MainNew:

    def __init__(self):
        self.spark = SparkSession.builder.appName("MLOps").getOrCreate()

    # Function to run all the functions
    def run(self):
        raw_data = func.loadCsvFile(spark=self.spark, file_name='HR-Employee-Dataset-Large.csv')
        df_clean = DataCleansing.dataCleaning(raw_data)
        qualify_data = func.checkMissingValues(df_clean)
        unqualified_columns = func.getMissingValueColumns(df_clean, 50)
        df_clean = func.dropColumns(df_clean, unqualified_columns)
        df_clean = func.missingValueImputation(df_clean)
        df_encode = DataCleansing.labelEncoding(df_clean)
        df_clean = DataCleansing.imputeOnce(df_encode)
        df_clean = Training.imbalanceAndModelTraining(df_clean)
        # # test.printCSV(df_clean)


if __name__ == '__main__':
    mainNew = MainNew()
    mainNew.run()
