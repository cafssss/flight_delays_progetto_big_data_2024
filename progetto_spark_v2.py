from pyspark.sql import SparkSession, Row
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, mean, count, when, lpad, concat, lit, \
                                    to_timestamp, month, year, sum as _sum, concat_ws, udf, \
                                    monotonically_increasing_id

from pyspark.ml.feature import Imputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from decouple import config
from graphframes import GraphFrame


def calcola_percentuale_valori_nulli(flights_df: DataFrame):
    # Contare i valori nulli in ciascuna colonna
    null_counts = flights_df.select(
        [count(when(col(c).isNull(), c)).alias(c) for c in flights_df.columns])

    # Calcolare la percentuale di valori nulli per ciascuna colonna
    total_count = flights_df.count()
    null_percentage = null_counts.select(
        [(col(c) / total_count * 100).alias(c) for c in null_counts.columns])
    null_percentage.show()


def imputer_colonne(df: DataFrame, nomi_colonne: list, strategia: str = "mean"):
    # Definire l'imputatore per sostituire i valori nulli con la media
    imputer = Imputer(inputCols=nomi_colonne,
                      outputCols=nomi_colonne).setStrategy(strategia)
    # Applicare l'imputatore
    df = imputer.fit(df).transform(df)
    return df


def drop_column(df, columns: list):

    # Cancellare le colonne specificate
    df = df.drop(*columns)

    return df


def divide_dataset(df):
    cancellati_df = df.filter(df["CANCELLED"] == 1)
    non_cancellati_df = df.filter(df["CANCELLED"] == 0)
    return cancellati_df, non_cancellati_df


def conc_date(df: DataFrame):
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

    # Mostrare il risultato
    # df.select("DEPARTURE").show(truncate=False)

    return df


def trasform_date(df: DataFrame, column: str):
    '''Funzione che trasforma una colonna da un formato HHMM in un formato HH:MM
    '''
    # Pad DEPARTURE_TIME con zeri se necessario
    df = df.withColumn(column, lpad(col(column), 4, '0'))

    # Dividerlo in ore e minuti
    df = df.withColumn("HOUR", col(column).substr(1, 2)) \
        .withColumn("MINUTE", col(column).substr(3, 2))

    # Creare una colonna temporanea combinando HOUR e MINUTE
    df = df.withColumn(column, concat(
        col("HOUR"), lit(":"), col("MINUTE"), lit(":00")))

    return df


def distribuzione_valori_nulli(df: DataFrame, column: str):
    # Aggiungere colonna di mese e anno dalla colonna data
    df = df.withColumn("Month", month(col("SCHEDULED_DEPARTURE")))
    df = df.withColumn("Year", year(col("SCHEDULED_DEPARTURE")))

    # Conteggio dei valori nulli
    df = df.withColumn("null_count", when(
        col(column).isNull(), 1).otherwise(0))

    # Aggregazione per mese e conteggio dei valori nulli
    monthly_null_counts = df.filter(col("null_count") == 1).groupBy("Year", "Month") \
                            .agg(count("null_count").alias("null_count")) \
                            .orderBy("Year", "Month")

    # Mostrare il risultato dell'aggregazione
    monthly_null_counts.show()

    # Convertire il risultato in un Pandas DataFrame per la visualizzazione con Matplotlib
    pandas_df = monthly_null_counts.toPandas()

    # Creare il grafico a barre
    plt.figure(figsize=(12, 6))
    plt.bar(pandas_df['Month'], pandas_df['null_count'], color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Number of Null Values')
    plt.title('Occurrences of Null Values by Month')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr',
               'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    return df


def delete_rows_null(df: DataFrame, column: list[str]):
    df = df.dropna(subset=column)
    return df


def most_delay(flights_df: DataFrame, airports_df: DataFrame):
    # Unione delle tabelle sulla colonna IATACODE
    joined_df = flights_df.join(
        airports_df, flights_df.ORIGIN_AIRPORT == airports_df.IATA_CODE, "inner")

    flights_count = joined_df.groupBy("CITY").agg(
        count("*").alias("NUM_FLIGHTS"))
    # Calcolare la media del numero di voli
    average_num_flights = flights_count.select(
        mean("NUM_FLIGHTS").alias("AVG_ROUTE"))

    # Arrotondo all'intero più vicino
    avg_flights = round(average_num_flights.collect()[0][0])

    # Filtrare le città con più della media di voli per rotta (in modo da filtrare le rotte più gettonate)
    filtered_routes = flights_count.filter(col("NUM_FLIGHTS") > avg_flights)

    # Unire il DataFrame originale con le città filtrate per ottenere i dettagli di ritardo solo per le città valide
    valid_df = joined_df.join(filtered_routes, on="CITY")

    # Calcolare la media dei ritardi di partenza e arrivo per ogni città
    average_delays = valid_df.groupBy("CITY") \
        .agg(mean("DEPARTURE_DELAY").alias("Average Departure Delay"),
             mean("ARRIVAL_DELAY").alias("Average Arrival Delay")).orderBy("Average Arrival Delay", ascending=False)

    average_delays = average_delays.withColumn(
        "TOTAL_AVG_DELAY",
        (col("Average Departure Delay") + col("Average Arrival Delay")) / 2
    )
    average_delays = average_delays.orderBy(col("TOTAL_AVG_DELAY").desc())
    # Convertire il risultato in un Pandas DataFrame per la visualizzazione con Matplotlib
    pandas_df = average_delays.toPandas()

    # Creare il grafico a barre
    plt.figure(figsize=(12, 8))

    # Numero di città da visualizzare
    num_cities = 20

    # Selezionare le prime num_cities città con i ritardi medi più alti di partenza
    top_cities = pandas_df.head(num_cities)

    # Costruire il grafico a barre
    bars1 = plt.barh(
        top_cities['CITY'], top_cities['Average Departure Delay'], color='b', label='Departure Delay')
    bars2 = plt.barh(top_cities['CITY'], top_cities['Average Arrival Delay'],
                     color='r', label='Arrival Delay', left=top_cities['Average Departure Delay'])

    # Aggiungere i numeri sul grafico
    for bar1, bar2 in zip(bars1, bars2):
        # Numeri al centro della barra blu
        plt.text(bar1.get_width() / 2, bar1.get_y() + bar1.get_height() / 2,
                 f'{bar1.get_width():.1f}', va='center', ha='center', color='white', fontsize=8, weight='bold')
        # Numeri alla fine della barra rossa
        plt.text(bar2.get_width() + bar1.get_width() + 0.2, bar2.get_y() + bar2.get_height() / 2,
                 f'{bar2.get_width():.1f}', va='center', ha='left', color='red', fontsize=8)

    # Aggiungere etichette, titolo e legenda
    plt.xlabel('Average Delay (minutes)')
    plt.ylabel('Cities')
    plt.title('Average Departure and Arrival Delays by City')
    plt.legend()

    # Mostrare il grafico
    plt.tight_layout()
    plt.show()
    '''    # Aggregare i ritardi per aeroporto
    delay_agg = df.select("CITY").distinct().groupBy("ORIGIN_AIRPORT").agg(
        mean("DEPARTURE_DELAY").alias("total_delay_start"),
        mean("ARRIVAL_DELAY").alias("total_delay_arrive")
    ).orderBy("total_delay_start", ascending=False)

    # Convertire il risultato in un Pandas DataFrame
    pandas_df = delay_agg.toPandas()

    # Creare il grafico a barre
    fig, ax = plt.subplots(figsize=(14, 7))

    # Posizione delle barre
    bar_width = 0.35
    index = pandas_df.index

    # Barre per delay_start
    bar1 = ax.bar(index, pandas_df['total_delay_start'],
                  bar_width, label='Delay Start')

    # Barre per delay_arrive
    bar2 = ax.bar(index + bar_width,
                  pandas_df['total_delay_arrive'], bar_width, label='Delay Arrive')

    # Aggiunta delle etichette e del titolo
    ax.set_xlabel('Origin Airports')
    ax.set_ylabel('Total Minutes of Delay')
    ax.set_title('Total Minutes of Delays by Airport')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(pandas_df['CITY'], rotation=45)
    ax.legend()

    # Mostrare il grafico
    plt.tight_layout()
    plt.show()'''

# n.8012839983


def graph_city_airline_delay(flights_df: DataFrame, airports_df: DataFrame, airlines_df: DataFrame):

    # Rinomina le colonne duplicate per evitare conflitti durante l'unione
    airports_df = airports_df.withColumnRenamed("AIRLINE", "AIRLINE_AIRPORTS")
    airlines_df = airlines_df.withColumnRenamed("AIRLINE", "AIRLINE_NAME")

    # Unione delle tabelle sulla colonna IATACODE
    joined_df = flights_df.join(airports_df, flights_df.ORIGIN_AIRPORT == airports_df.IATA_CODE, "inner") \
        .join(airlines_df, flights_df.AIRLINE == airlines_df.IATA_CODE, "inner")

    flights_count = joined_df.groupBy("CITY").agg(
        count("*").alias("NUM_FLIGHTS"))
    # Calcolare la media del numero di voli
    average_num_flights = flights_count.select(
        mean("NUM_FLIGHTS").alias("AVG_ROUTE"))

    # Arrotondo all'intero più vicino
    avg_flights = round(average_num_flights.collect()[0][0])

    # Filtrare le città con più della media di voli per rotta (in modo da filtrare le rotte più gettonate)
    filtered_routes = flights_count.orderBy("NUM_FLIGHTS", ascending=False).limit(10)

    # Unire il DataFrame originale con le città filtrate per ottenere i dettagli di ritardo solo per le città valide
    valid_df = joined_df.join(filtered_routes, on="CITY")
    # Calcolare la media dei ritardi di partenza per città e compagnia aerea
    average_delays = valid_df.groupBy("CITY", "AIRLINE_NAME") \
        .agg(mean("DEPARTURE_DELAY").alias("Average Departure Delay")).orderBy("Average Departure Delay", ascending=False)\
        # .orderBy("Average Departure Delay", ascending=False)

    # Convertire il risultato in un Pandas DataFrame per la visualizzazione con Seaborn
    pandas_df = average_delays.toPandas()

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Average Departure Delay', y='CITY', hue='AIRLINE_NAME', data=pandas_df)
    plt.title('Media dei minuti di ritardo per città e compagnia Aerea')
    plt.xlabel('Media dei minuti di ritardo')
    plt.ylabel('Città')
    plt.legend(title='Compagnia Aerea', fontsize='small')
    plt.show()


'''    # Pivot della tabella per creare la matrice per la heatmap
    heatmap_data = pandas_df.pivot(
        index="CITY", columns="AIRLINE_NAME", values="Average Departure Delay")

    # Creare la heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data, cmap=sns.color_palette(
        "flare", as_cmap=True), annot=False, vmax=30, vmin=0)

    # Aggiungere etichette, titolo e personalizzazione
    plt.title('Delays: impact of the origin airport')
    plt.xlabel('Airlines')
    plt.ylabel('Cities')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.ylim(0, 230)
    # Aumentare la spaziatura per le etichette degli assi
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Mostrare il grafico
    plt.show()'''


def heatmap_delay_dinamic(flights_df: DataFrame, airports_df: DataFrame, airlines_df: DataFrame):

    # Rinomina le colonne duplicate per evitare conflitti durante l'unione
    airports_df = airports_df.withColumnRenamed("AIRLINE", "AIRLINE_AIRPORTS")
    airlines_df = airlines_df.withColumnRenamed("AIRLINE", "AIRLINE_NAME")

    # Unione delle tabelle sulla colonna IATACODE
    joined_df = flights_df.join(airports_df, flights_df.ORIGIN_AIRPORT == airports_df.IATA_CODE, "inner") \
        .join(airlines_df, flights_df.AIRLINE == airlines_df.IATA_CODE, "inner")

    # Calcolare la media dei ritardi di partenza per città e compagnia aerea
    average_delays = joined_df.groupBy("CITY", "AIRLINE_NAME") \
        .agg(mean("DEPARTURE_DELAY").alias("Average Departure Delay"))\
        # .orderBy("Average Departure Delay", ascending=False)

    # Convertire il risultato in un Pandas DataFrame per la visualizzazione con Seaborn
    pandas_df = average_delays.toPandas()

    # Creare la heatmap interattiva con plotly
    fig = px.imshow(pandas_df.pivot(index="CITY", columns="AIRLINE_NAME", values="Average Departure Delay"),
                    labels=dict(x="Airlines", y="Cities",
                                color="Average Departure Delay"),
                    x=pandas_df['AIRLINE_NAME'].unique(),
                    y=pandas_df['CITY'].unique(),
                    aspect="auto",
                    zmax=30,
                    zmin=0,
                    color_continuous_scale=[[0, 'yellow'], [0.5, 'orange'], [1.0, 'red']])

    # Aggiungere il titolo
    fig.update_layout(title="Delays: impact of the origin airport")

    # Mostrare il grafico
    fig.show()


#  CANCELLAZIONI VOLI
def plot_cancellation_reasons(cancelled_flights_df: DataFrame):

    # Raggruppa e conta i voli cancellati per motivo della cancellazione e li ordina in modo decrescente
    cancellation_reasons_df = cancelled_flights_df.groupBy('CANCELLATION_REASON') \
        .agg(count('*').alias('count')) \
        .orderBy('count', ascending=False) \
        .toPandas()

    # Cause cancellazione volo
    cancellation_reason_map = {
        'A': 'Airline',
        'B': 'Weather',
        'C': 'Air Traffic Control',
        'D': 'Security'
    }

    # Sostituire letteere con le motivazioni del ritardo
    cancellation_reasons_df['CANCELLATION_REASON'] = cancellation_reasons_df['CANCELLATION_REASON'].map(
        cancellation_reason_map)

    plt.figure(figsize=(10, 6))

    # Creazione del grafico a barre
    bars = plt.bar(cancellation_reasons_df['CANCELLATION_REASON'],
                   cancellation_reasons_df['count'], color='royalblue')

    # Etichette
    plt.xlabel('Cancellation Reason')
    plt.ylabel('Number of Cancellations')

    # Aggiunta titolo
    plt.title('Main Reasons for Flight Cancellations')

    # Aggiungere i numeri delle cancellazioni sopra le barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 10,
                 int(yval), ha='center', va='bottom', fontsize=11)

    # Mostrare il grafico
    plt.show()


def plot_cancellations_by_month(cancelled_flights_df: DataFrame):

    # Raggruppa ed ordina per mese (January = 1, ..., December = 12)
    cancellations_count = cancelled_flights_df.groupBy('MONTH').count()
    cancellations_count = cancellations_count.sort('MONTH')
    cancellations_pd = cancellations_count.toPandas()

    # Plot
    plt.figure(figsize=(10, 6))

    # Creazione grafico a barre
    plt.bar(cancellations_pd['MONTH'],
            cancellations_pd['count'], color='lightcoral')

    # Titolo
    plt.title('Number of cancelled flights per month')

    # Etichette
    plt.xlabel('Month')
    plt.ylabel('Number of cancelled flights')

    # Mesi
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(ticks=range(0, 12), labels=months)

    # Riga che parte dall'asse delle ordinate
    plt.grid(axis='y')
    plt.show()


def plot_cancellations_by_day_of_week(cancelled_flights_df: DataFrame):

    cancellations_count = cancelled_flights_df.groupBy('DAY_OF_WEEK').count()

    # Ordina i risultati per il giorno della settimana (Monday = 1, ..., Sunday = 7)
    cancellations_count = cancellations_count.sort('DAY_OF_WEEK')

    cancellations_pd = cancellations_count.toPandas()

    # Plot
    plt.figure(figsize=(10, 6))

    # Creazione del grafico a barre
    plt.bar(cancellations_pd['DAY_OF_WEEK'],
            cancellations_pd['count'], color='skyblue')

    # Titolo
    plt.title('Number of cancelled flights per day of the week')

    # Etichette
    plt.xlabel('Day of the week')
    plt.ylabel('Number of cancelled flights')

    # Giorni della settimana
    plt.xticks(range(1, 8), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Riga che parte dall'asse delle ordinate
    plt.grid(axis='y')
    plt.show()


def route_most_delay(df_flights: DataFrame, airport_df: DataFrame):
    '''    df_flights = df_flights.join(
        airport_df,
        df_flights.ORIGIN_AIRPORT == airport_df.IATA_CODE,
        "left"
    ).select(
        df_flights["*"],
        col("AIRPORT").alias("AIRPORT_ORIGIN")
    )

    # Join del risultato precedente con la tabella airports per ottenere la città di destinazione
    df_flights = df_flights.join(
        airport_df,
        df_flights.DESTINATION_AIRPORT == airport_df.IATA_CODE,
        "left"
    ).select(
        df_flights["*"],
        col("AIRPORT").alias("AIRPORT_DESTINATION")
    )
'''
    df_flights = df_flights.withColumn("ROUTE", concat_ws(
        "-", df_flights.ORIGIN_AIRPORT, df_flights.DESTINATION_AIRPORT))

    # Calcolare il numero di voli per ciascuna rotta
    route_counts = df_flights.groupBy("ROUTE").count()

    # Calcolare la media del numero di voli
    average_delay_route = route_counts.select(mean("count").alias("AVG_ROUTE"))

    # Arrotondo all'intero più vicino
    avg_route = round(average_delay_route.collect()[0][0])

    # Filtrare le rotte con più della media di voli per rotta (in modo da filtrare le rotte più gettonate)
    filtered_routes = route_counts.filter(col("count") > avg_route)

    # Unire il DataFrame originale con le rotte filtrate per ottenere i dettagli di ritardo solo per le rotte valide
    valid_routes = df_flights.join(filtered_routes, on="ROUTE")

    # Calcolare il ritardo medio per ciascuna rotta
    avg_delays = valid_routes.groupBy("ROUTE").agg(
        mean("DEPARTURE_DELAY").alias("AVG_DEPARTURE_DELAY"),
        mean("ARRIVAL_DELAY").alias("AVG_ARRIVAL_DELAY")
    )

    # Calcolare il ritardo totale medio combinato (per un singolo valore di ritardo medio per la rotta)
    avg_delays = avg_delays.withColumn(
        "TOTAL_AVG_DELAY",
        (col("AVG_DEPARTURE_DELAY") + col("AVG_ARRIVAL_DELAY")) / 2
    )

    # Ordinare le rotte per ritardo medio in ordine decrescente
    sorted_routes = avg_delays.orderBy(col("TOTAL_AVG_DELAY").desc())

    # Convertire il DataFrame PySpark in Pandas DataFrame
    sorted_routes_pandas = sorted_routes.toPandas()

    # Creare il grafico a barre
    fig = px.bar(sorted_routes_pandas.head(10), x="ROUTE", y="TOTAL_AVG_DELAY",
                 labels={
                     "TOTAL_AVG_DELAY": "Average Total Delay (minutes)", "ROUTE": "Route"},
                 title="Top 10 Routes with Highest Average Delays")

    # Mostrare il grafico
    fig.show()


def most_delay_airport_most_flight(flights: DataFrame):
    # Calcolare il numero di voli per ogni aeroporto di partenza
    flights_count = flights.groupBy("ORIGIN_AIRPORT").agg(
        count("*").alias("NUM_FLIGHTS"))

    # Calcolare il ritardo medio di partenza per ogni aeroporto
    avg_departure_delay = flights.groupBy("ORIGIN_AIRPORT").agg(
        mean("DEPARTURE_DELAY").alias("AVG_DEPARTURE_DELAY"))

    flights_analysis = flights_count.join(
        avg_departure_delay, on="ORIGIN_AIRPORT")

    flights_analysis = flights_analysis.orderBy(
        "AVG_DEPARTURE_DELAY", ascending=False)

    # Convertire il DataFrame PySpark in un Pandas DataFrame
    flights_analysis_pandas = flights_analysis.toPandas()

    # Creare il grafico a barre
    fig = px.scatter(flights_analysis_pandas, x="NUM_FLIGHTS", y="AVG_DEPARTURE_DELAY",
                     labels={"NUM_FLIGHTS": "Number of Flights",
                             "AVG_DEPARTURE_DELAY": "Average Departure Delay (minutes)"},
                     title="Relationship between Number of Flights and Average Departure Delay at Airports")

    # Mostrare il grafico
    fig.show()


def origin_airport_pie(flights: DataFrame, airports: DataFrame, spark):
    # Fare la join tra flights e airports per ottenere i nomi degli aeroporti
    flights_with_airport_names = flights.join(
        airports,
        flights.ORIGIN_AIRPORT == airports.IATA_CODE,
        "inner"
    ).select(
        flights["*"],
        airports["AIRPORT"].alias("ORIGIN_AIRPORT_NAME")
    )

    # Calcolare il numero di voli per ogni aeroporto di origine
    flights_count = flights_with_airport_names.groupBy(
        "ORIGIN_AIRPORT_NAME").agg(count("*").alias("NUM_FLIGHTS"))

    # Calcolare il totale dei voli
    total_flights = flights_count.agg(_sum("NUM_FLIGHTS").alias(
        "TOTAL_FLIGHTS")).collect()[0]["TOTAL_FLIGHTS"]
    print(total_flights)
    # Calcolare la percentuale di voli per ogni aeroporto
    flights_percentage = flights_count.withColumn(
        "PERCENTAGE", (col("NUM_FLIGHTS") / total_flights) * 100)

    flights_percentage_first = flights_percentage.orderBy(
        "PERCENTAGE", ascending=False).limit(15)

    # df_without_first = flights_percentage.exceptAll(flights_percentage_first)

    # percentage_other_airport = df_without_first.agg(_sum("PERCENTAGE").alias("PERCENTAGE_LAST")).collect()[0]["PERCENTAGE_LAST"]

    # total_flights_last = df_without_first.agg(_sum("NUM_FLIGHTS").alias("TOTAL_FLIGHTS")).collect()[0]["TOTAL_FLIGHTS"]

    # n_other_airport = df_without_first.count()

    # newRow = Row(NUM_FLIGHTS=total_flights_last, ORIGIN_AIRPORT_NAME="Other airport", PERCENTAGE=percentage_other_airport)
    # flights_percentage_first = flights_percentage_first.union([newRow])
    # Convertire il DataFrame PySpark in un Pandas DataFrame
    flights_pandas = flights_percentage_first.toPandas()

    # Creare il grafico a torta
    fig = px.pie(flights_pandas, values="NUM_FLIGHTS", names="ORIGIN_AIRPORT_NAME",
                 title="Percentage of Flights by Origin Airport")

    # Mostrare il grafico
    fig.show()


def graph_cities_interconnected(flights_df: DataFrame, airports_df: DataFrame):
    # Creare il DataFrame dei vertici
    vertices = airports_df.withColumnRenamed("IATA_CODE", "id")

    # Creare il DataFrame degli archi
    edges = flights_df.withColumnRenamed(
        "ORIGIN_AIRPORT", "src").withColumnRenamed("DESTINATION_AIRPORT", "dst")

    # Creare il grafo
    graph = GraphFrame(vertices, edges)

    # Calcolare il numero di connessioni in entrata per ciascuna città
    in_degrees = graph.inDegrees


    # Ordinare per numero di connessioni in entrata e mostrare le città più connesse
    in_degrees= in_degrees.select("id", "inDegree").orderBy("inDegree", ascending=False).limit(15)

    # Eseguire l'algoritmo PageRank
    pagerank_results = graph.pageRank(resetProbability=0.15, maxIter=10)
    pagerank_results = pagerank_results.vertices.select("id", "pagerank").orderBy("pagerank", ascending=False).limit(15)

    triangle_count = graph.triangleCount().select("id", "count").orderBy("count", ascending=False).limit(15)

    # Mostrare i risultati
    in_degrees.show()
    pagerank_results.show()
    triangle_count.show()

    # Convertire i risultati in Pandas DataFrame
    pagerank_df = pagerank_results.toPandas()
    triangle_count_df = triangle_count.toPandas()
    in_degrees = in_degrees.toPandas()
  
    # Configurazione dello stile di Seaborn
    sns.set(style="whitegrid")

    # Visualizzare il PageRank
    plt.figure(figsize=(10, 6))
    sns.barplot(x="id", y="pagerank", data=pagerank_df)
    plt.title("PageRank degli Aeroporti")
    plt.ylabel("PageRank")
    plt.xlabel("Aeroporto")
    plt.xticks(rotation=45)
    plt.show()

    # Visualizzare il conteggio dei triangoli
    plt.figure(figsize=(10, 6))
    sns.barplot(x="id", y="count", data=triangle_count_df)
    plt.title("Conteggio dei Triangoli per Aeroporto")
    plt.ylabel("Numero di Triangoli")
    plt.xlabel("Aeroporto")
    plt.xticks(rotation=45)
    plt.show()

    # Visualizzare il grado di connessione
    plt.figure(figsize=(10, 6))
    sns.barplot(x="id", y="inDegree", data=in_degrees)
    plt.title("Grado di Connessione degli Aeroporti")
    plt.ylabel("Grado")
    plt.xlabel("Aeroporto")
    plt.xticks(rotation=45)
    plt.show()


#################################################################################


def main():
    # Creazione della sessione Spark
    spark = SparkSession.builder \
        .appName("Flight Delays and Cancellations Analysis") \
        .config("spark.driver.memory", "4g")\
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2")\
        .config("spark.executor.instances", "4") \
        .getOrCreate()

    columns_to_drop_init = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR',
                            'DAY', 'DATE', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
                            'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DIVERTED', 'FLIGHT_NUMBER',
                            'TAIL_NUMBER', 'AIR_TIME', 'HOUR', 'MINUTE']
    # Percorso ai file CSV
    file_path_flights = config("PATH_FLIGHTS")
    file_path_airport = config("PATH_AIRPORT")
    file_path_airline = config("PATH_AIRLINE")
    # Caricamento dei dataset
    flights_df = spark.read.csv(
        file_path_flights, header=True, inferSchema=True)
    airports_df = spark.read.csv(
        file_path_airport, header=True, inferSchema=True)
    airlines_df = spark.read.csv(
        file_path_airline, header=True, inferSchema=True)
    # Unisce le colonne che rappresentano anno, mese, giorno e orario di partenza programmata in una sola colonna
    flights_df = conc_date(flights_df)
    # Trasforma l'orario del dataset in un formato più leggibile (HH:MM)
    flights_df = trasform_date(flights_df, "DEPARTURE_TIME")
    flights_df = trasform_date(flights_df, "SCHEDULED_ARRIVAL")
    flights_df = trasform_date(flights_df, "ARRIVAL_TIME")
    # Elimina le colonne che non servono
    flights_df = drop_column(flights_df, columns_to_drop_init)
    # Divide in due i dataset
    voli_cancellati, voli_in_ritardo = divide_dataset(flights_df)

    columns_to_drop_cancellati = ['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ELAPSED_TIME',
                                  'ARRIVAL_TIME', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']

    columns_to_drop_ritardo = ['CANCELLED', 'CANCELLATION_REASON']

    voli_cancellati = drop_column(voli_cancellati, columns_to_drop_cancellati)
    voli_in_ritardo = drop_column(voli_in_ritardo, columns_to_drop_ritardo)

    # eliminati 15187 su 5729195 equivalente al 0,27% si può optare pure un imputazione
    voli_in_ritardo = delete_rows_null(voli_in_ritardo, ["ELAPSED_TIME"])

    graph_cities_interconnected(voli_in_ritardo, airports_df)
    # df=distribuzione_valori_nulli(voli_in_ritardo, "ARRIVAL_TIME")
    origin_airport_pie(voli_in_ritardo, airports_df, spark)
    # funzione che mostra in un grafico le citta con più ritardo in ingresso e uscita
    most_delay(voli_in_ritardo, airports_df)

    # funzione che disegna un grafico fra città e compagnie,
    # i valori rappresentano invece la media dei minuti di ritardo per compagnia
    # le città visualizzate sono le città che hanno più voli
    graph_city_airline_delay(voli_in_ritardo, airports_df, airlines_df)

    # funzione che mostra le principali motivazioni delle cancellazioni dei voli
    plot_cancellation_reasons(voli_cancellati)

    # Funzione che mostra il numero di cancellazioni suddivise per mesi
    plot_cancellations_by_month(voli_cancellati)

    # Funzione che mostra il numero di cancellazioni suddivise per giorno della settimana
    plot_cancellations_by_day_of_week(voli_cancellati)

    # [partenza, arrivo] con più minuti di ritardi
    route_most_delay(voli_in_ritardo, airports_df)

    # Negli aeroporti con più voli ci sono anche più ritardi?
    most_delay_airport_most_flight(voli_in_ritardo)

    print(0)


if __name__ == "__main__":
    main()
