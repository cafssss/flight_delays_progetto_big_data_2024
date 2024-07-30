from pyspark.sql import SparkSession, Row
from decouple import config
from src.preprocessing import PreprocessingFlights
from src.delayed_flights import DelayedFlights
from src.cancelled_flights import CancelledFlights
from src.graphframes_flights import GraphframeFlights
import warnings


def main():

    warnings.filterwarnings("ignore")

    # Creazione della sessione Spark
    spark = SparkSession.builder \
        .appName("Flight Delays and Cancellations Analysis") \
        .config("spark.driver.memory", "4g")\
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2")\
        .config("spark.executor.instances", "4") \
        .getOrCreate()

    # Caricamento dei dataset
    flights_df = spark.read.csv(
        config("PATH_FLIGHTS"), header=True, inferSchema=True)
    airports_df = spark.read.csv(
        config("PATH_AIRPORT"), header=True, inferSchema=True)
    airlines_df = spark.read.csv(
        config("PATH_AIRLINE"), header=True, inferSchema=True)

    # Istanzio un oggetto e richiamo un metodo che permette il preprocessing del dataset
    preprocessing = PreprocessingFlights(flights_df)
    cancelled_flights, delayed_flights = preprocessing.preprocessing_data()

    # Istanzio un oggetto e richiamo dei metodi per l'analisi dei voli in ritardo
    d = DelayedFlights(delayed_flights, airports_df, airlines_df)
    d.origin_airport_pie()
    d.most_delay()
    d.avg_delay()
    d.delay_analysis()
    d.graph_city_airline_delay()
    d.most_delay_airport_most_flight()
    d.route_most_delay()
    d.plot_cities_with_most_delays()

    # Istanzio un oggetto e richiamo dei metodi per l'analisi dei voli cancellati
    c = CancelledFlights(cancelled_flights)
    c.plot_cancellation_reasons()
    c.plot_cancellation_reasons_pie()
    c.cancellation_per_airline()
    c.plot_cancellations_by_month()
    c.plot_cancellations_by_day_of_week()
    c.cancellations_per_airport()

    # Istanzio un oggetto e richiamo dei metodi per l'analisi dei grafi
    g = GraphframeFlights(delayed_flights, airports_df)
    g.graph_cities_interconnected()
    g.graph_states_interconnected()


if __name__ == "__main__":
    main()