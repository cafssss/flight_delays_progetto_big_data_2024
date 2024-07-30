from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, mean, count, when, lpad, concat, lit, \
                                    to_timestamp, month, year, sum as _sum, concat_ws



class PreprocessingFlights:

    def __init__(self, df: DataFrame):
        self.df_flights = df
        self.cancelled_flights = None
        self.delayed_flights = None

    def __percentage_of_null_values(self, df: DataFrame):
        # Contare i valori nulli in ciascuna colonna
        null_counts = df.select(
            [count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

        # Calcolare la percentuale di valori nulli per ciascuna colonna
        total_count = df.count()
        null_percentage = null_counts.select(
            [(col(c) / total_count * 100).alias(c) for c in null_counts.columns])
        null_percentage.show()

    def __conc_date(self):
        ''' Questa funzione concatena le date
        '''
        df = self.df_flights
        # Pad MONTH e DAY con zeri se necessario
        df = df.withColumn("MONTH", lpad(col("MONTH"), 2, '0')) \
            .withColumn("DAY", lpad(col("DAY"), 2, '0'))

        # Pad SCHEDULED_DEPARTURE con zeri se necessario e poi dividerlo in ore e minuti
        df = df.withColumn("SCHEDULED_DEPARTURE", lpad(
            col("SCHEDULED_DEPARTURE"), 4, '0'))
        df = df.withColumn("HOUR", col("SCHEDULED_DEPARTURE").substr(1, 2)) \
            .withColumn("MINUTE", col("SCHEDULED_DEPARTURE").substr(3, 2))

        # Modificare la colonna SCHEDULED_DEPARTURE combinando YEARS, MONTH, DAY, HOUR e MINUTE
        df = df.withColumn("SCHEDULED_DEPARTURE", concat(col("YEAR"), lit("-"),
                                                        col("MONTH"), lit("-"),
                                                        col("DAY"), lit(" "),
                                                        col("HOUR"), lit(":"),
                                                        col("MINUTE"), lit(":00")))

        # Convertire la colonna SCHEDULED_DEPARTURE in timestamp
        df = df.withColumn("SCHEDULED_DEPARTURE", to_timestamp(
            col("SCHEDULED_DEPARTURE"), "yyyy-MM-dd HH:mm:ss"))
        
        self.df_flights = df 

    def __trasform_date(self, column: str):
        '''Funzione che trasforma una colonna da un formato HHMM in un formato HH:MM
        '''
        df = self.df_flights
        # Pad DEPARTURE_TIME con zeri se necessario
        df = df.withColumn(column, lpad(col(column), 4, '0'))

        # Dividerlo in ore e minuti
        df = df.withColumn("HOUR", col(column).substr(1, 2)) \
            .withColumn("MINUTE", col(column).substr(3, 2))

        # Creare una colonna temporanea combinando HOUR e MINUTE
        df = df.withColumn(column, concat(
            col("HOUR"), lit(":"), col("MINUTE"), lit(":00")))
        
        self.df_flights = df 

    def __divide_dataset(self):
        df = self.df_flights
        self.cancelled_flights = df.filter(df["CANCELLED"] == 1)
        self.delayed_flights = df.filter(df["CANCELLED"] == 0)
    
    def __preprocessing_original_flights(self):
        columns_to_drop_init = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR',
                        'DAY', 'DATE', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
                        'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DIVERTED', 'FLIGHT_NUMBER',
                        'TAIL_NUMBER', 'AIR_TIME', 'HOUR', 'MINUTE']
        self.__conc_date()
        self.__trasform_date("DEPARTURE_TIME")
        self.__trasform_date("SCHEDULED_ARRIVAL")
        self.__trasform_date("ARRIVAL_TIME")

        # Elimina le colonne che non servono per l'analisi
        self.df_flights = self.df_flights.drop(*columns_to_drop_init)
        self.__divide_dataset()
    
    def __preprocessing_cancelled_flights(self):
        self.__percentage_of_null_values(self.cancelled_flights)
        columns_to_drop_cancelled = ['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ELAPSED_TIME',
                                'ARRIVAL_TIME', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']
        # Elimina le colonne che non servono per l'analisi
        self.cancelled_flights = self.cancelled_flights.drop(*columns_to_drop_cancelled)

        self.__percentage_of_null_values(self.cancelled_flights)

    def __preprocessing_delayed_flights(self):
        self.__percentage_of_null_values(self.delayed_flights)
        columns_to_drop_delayed = ['CANCELLED', 'CANCELLATION_REASON']

        # Elimina le colonne che non servono per l'analisi
        self.delayed_flights = self.delayed_flights.drop(*columns_to_drop_delayed)
        # Elimina le righe null della colonna specificata in subset
        self.delayed_flights = self.delayed_flights.dropna(subset=["ELAPSED_TIME"])

        self.__percentage_of_null_values(self.delayed_flights)

    def preprocessing_data(self):
        self.__preprocessing_original_flights()
        self.__preprocessing_cancelled_flights()
        self.__preprocessing_delayed_flights()
        return self.cancelled_flights, self.delayed_flights

'''
def main():
    p = FlightsPreprocessing(df_flight)
    p.preprocessing_data()
'''