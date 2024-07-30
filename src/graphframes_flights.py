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

class GraphframeFlights:

    def __init__(self, delayed_flights: DataFrame, airports_df: DataFrame):
        self.delayed_flights = delayed_flights
        self.airports_df = airports_df
    
    def graph_cities_interconnected(self):
        
        # Creare il DataFrame dei vertici
        vertices = self.airports_df.withColumnRenamed("IATA_CODE", "id")

        # Creare il DataFrame degli archi
        edges = self.delayed_flights.withColumnRenamed(
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
        sns.set_theme(style="whitegrid")

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
