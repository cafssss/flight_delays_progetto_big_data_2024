#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, mean, count, when, lpad, concat, lit, to_timestamp, month, year, sum, coalesce
from pyspark.ml.feature import Imputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


def calcola_percentuale_valori_nulli(flights_df: DataFrame):
    # Contare i valori nulli in ciascuna colonna
    null_counts = flights_df.select(
        [count(when(col(c).isNull(), c)).alias(c) for c in flights_df.columns])

    # Calcolare la percentuale di valori nulli per ciascuna colonna
    total_count = flights_df.count()
    null_percentage = null_counts.select(
        [(col(c) / total_count * 100).alias(c) for c in null_counts.columns])
    null_percentage.show()


# In[3]:


def imputer_colonne(df:DataFrame ,nomi_colonne: list, strategia: str = "mean"):
    # Definire l'imputatore per sostituire i valori nulli con la media
    imputer = Imputer(inputCols=nomi_colonne,
                      outputCols=nomi_colonne).setStrategy(strategia)
    # Applicare l'imputatore
    df = imputer.fit(df).transform(df)
    return df


# In[4]:


def drop_column(df, columns:list):

    # Cancellare le colonne specificate
    df = df.drop(*columns)

    return df


# In[5]:


def divide_dataset(df):
    cancellati_df = df.filter(df["CANCELLED"] == 1)
    non_cancellati_df = df.filter(df["CANCELLED"] == 0)
    return cancellati_df, non_cancellati_df


# In[6]:


def conc_date(df: DataFrame):
    # Pad MONTH e DAY con zeri se necessario
    df = df.withColumn("MONTH", lpad(col("MONTH"), 2, '0')).withColumn("DAY", lpad(col("DAY"), 2, '0'))

    # Pad SCHEDULED_DEPARTURE con zeri se necessario e poi dividerlo in ore e minuti
    df = df.withColumn("SCHEDULED_DEPARTURE", lpad(col("SCHEDULED_DEPARTURE"), 4, '0'))
    df = df.withColumn("HOUR", col("SCHEDULED_DEPARTURE").substr(1, 2)).withColumn("MINUTE", col("SCHEDULED_DEPARTURE").substr(3, 2))

    # Modificare la colonna SCHEDULED_DEPARTURE combinando YEARS, MONTH, DAY, HOUR e MINUTE
    df = df.withColumn("SCHEDULED_DEPARTURE", concat(col("YEAR"), lit("-"),
                                        col("MONTH"), lit("-"),
                                        col("DAY"), lit(" "),
                                        col("HOUR"), lit(":"),
                                        col("MINUTE"), lit(":00")))

    # Convertire la colonna SCHEDULED_DEPARTURE in timestamp
    df = df.withColumn("SCHEDULED_DEPARTURE", to_timestamp(col("SCHEDULED_DEPARTURE"), "yyyy-MM-dd HH:mm:ss"))

    # Mostrare il risultato
    #df.select("DEPARTURE").show(truncate=False)
    
    return df


# In[7]:


def trasform_date(df: DataFrame, column: str):
    '''Funzione che trasforma una colonna da un formato HHMM in un formato HH:MM
    '''
    # Pad DEPARTURE_TIME con zeri se necessario
    df = df.withColumn(column, lpad(col(column), 4, '0'))

    # Dividerlo in ore e minuti
    df = df.withColumn("HOUR", col(column).substr(1, 2))         .withColumn("MINUTE", col(column).substr(3, 2))
        
    # Creare una colonna temporanea combinando HOUR e MINUTE 
    df = df.withColumn(column, concat(col("HOUR"), lit(":"), col("MINUTE"), lit(":00")))
    
    return df


# In[8]:


def distribuzione_valori_nulli(df: DataFrame, column:str):
    # Aggiungere colonna di mese e anno dalla colonna data
    df = df.withColumn("Month", month(col("SCHEDULED_DEPARTURE")))
    df = df.withColumn("Year", year(col("SCHEDULED_DEPARTURE")))

    # Conteggio dei valori nulli
    df = df.withColumn("null_count", when(col(column).isNull(), 1).otherwise(0))


    # Aggregazione per mese e conteggio dei valori nulli
    monthly_null_counts = df.filter(col("null_count")==1).groupBy("Year", "Month")                             .agg(count("null_count").alias("null_count"))                             .orderBy("Year", "Month")

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
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    return df


# In[9]:


'''
def delete_rows_null(df: DataFrame, column:list[str]): 
    df= df.dropna(subset=column)
    return df
'''


# In[10]:


def most_delay(flights_df: DataFrame, airports_df: DataFrame):
    # Unione delle tabelle sulla colonna IATACODE
    joined_df = flights_df.join(airports_df, flights_df.ORIGIN_AIRPORT == airports_df.IATA_CODE, "inner")

    # Calcolare la media dei ritardi di partenza e arrivo per ogni città
    average_delays = joined_df.groupBy("CITY")                             .agg(mean("DEPARTURE_DELAY").alias("Average Departure Delay"),
                                mean("ARRIVAL_DELAY").alias("Average Arrival Delay")).orderBy("Average Departure Delay", ascending=False)

    # Convertire il risultato in un Pandas DataFrame per la visualizzazione con Matplotlib
    pandas_df = average_delays.toPandas()

    # Creare il grafico a barre
    plt.figure(figsize=(12, 8))

    # Numero di città da visualizzare
    num_cities = 50

    # Selezionare le prime num_cities città con i ritardi medi più alti di partenza
    top_cities = pandas_df.head(num_cities)

    # Costruire il grafico a barre
    bars1 = plt.barh(top_cities['CITY'], top_cities['Average Departure Delay'], color='b', label='Departure Delay')
    bars2 = plt.barh(top_cities['CITY'], top_cities['Average Arrival Delay'], color='r', label='Arrival Delay', left=top_cities['Average Departure Delay'])
    
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
    bar1 = ax.bar(index, pandas_df['total_delay_start'], bar_width, label='Delay Start')

    # Barre per delay_arrive
    bar2 = ax.bar(index + bar_width, pandas_df['total_delay_arrive'], bar_width, label='Delay Arrive')

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


# In[11]:


def delays_per_airport(flights_df: DataFrame, airports_df: DataFrame):
    # Calcola i ritardi totali in partenza per ogni aeroporto
    departure_delays = flights_df.groupBy('ORIGIN_AIRPORT').agg(
        sum('DEPARTURE_DELAY').alias('total_departure_delay')
    )
    
    # Calcola i ritardi totali in arrivo per ogni aeroporto
    arrival_delays = flights_df.groupBy('DESTINATION_AIRPORT').agg(
        sum('ARRIVAL_DELAY').alias('total_arrival_delay')
    )
    
    # Assegna alias ai DataFrame
    departure_delays = departure_delays.alias('dep')
    arrival_delays = arrival_delays.alias('arr')
    airports_df = airports_df.alias('air')
    
    # Unisce i ritardi di partenza e arrivo per ciascun aeroporto
    total_delays = departure_delays.join(
        arrival_delays,
        col('dep.ORIGIN_AIRPORT') == col('arr.DESTINATION_AIRPORT'),
        how='outer'
    ).select(
        coalesce(col('dep.ORIGIN_AIRPORT'), col('arr.DESTINATION_AIRPORT')).alias('AIRPORT'),
        col('total_departure_delay'),
        col('total_arrival_delay')
    )
    
    # Somma i ritardi di partenza e arrivo
    total_delays = total_delays.withColumn(
        'total_delay',
        coalesce(col('total_departure_delay'), lit(0)) + coalesce(col('total_arrival_delay'), lit(0))
    ).na.fill(0)  # Riempi i valori nulli con 0
    
    # Unisce con il DataFrame degli aeroporti per ottenere i nomi completi
    total_delays = total_delays.join(
        airports_df,
        total_delays.AIRPORT == col('air.IATA_CODE'),
        how='left'
    ).select(
        col('AIRPORT'),
        col('air.AIRPORT').alias('AIRPORT_NAME'),
        col('total_departure_delay'),
        col('total_arrival_delay'),
        col('total_delay')
    )
    
    return total_delays


# In[12]:


def cities_with_most_delays(flights_df: DataFrame, airports_df: DataFrame):
    # Calcola i ritardi totali in partenza per ogni aeroporto
    departure_delays = flights_df.groupBy('ORIGIN_AIRPORT').agg(
        sum('DEPARTURE_DELAY').alias('total_departure_delay')
    )
    
    # Calcola i ritardi totali in arrivo per ogni aeroporto
    arrival_delays = flights_df.groupBy('DESTINATION_AIRPORT').agg(
        sum('ARRIVAL_DELAY').alias('total_arrival_delay')
    )
    
    # Assegna alias ai DataFrame
    departure_delays = departure_delays.alias('dep')
    arrival_delays = arrival_delays.alias('arr')
    airports_df = airports_df.alias('air')
    
    # Unisce i ritardi di partenza e arrivo per ciascun aeroporto
    total_delays = departure_delays.join(
        arrival_delays,
        col('dep.ORIGIN_AIRPORT') == col('arr.DESTINATION_AIRPORT'),
        how='outer'
    ).select(
        coalesce(col('dep.ORIGIN_AIRPORT'), col('arr.DESTINATION_AIRPORT')).alias('AIRPORT'),
        col('total_departure_delay'),
        col('total_arrival_delay')
    )
    
    # Somma i ritardi di partenza e arrivo
    total_delays = total_delays.withColumn(
        'total_delay',
        coalesce(col('total_departure_delay'), lit(0)) + coalesce(col('total_arrival_delay'), lit(0))
    ).na.fill(0)  # Riempi i valori nulli con 0
    
    # Unisce con il DataFrame degli aeroporti per ottenere le città
    total_delays = total_delays.join(
        airports_df,
        total_delays.AIRPORT == col('air.IATA_CODE'),
        how='left'
    ).select(
        col('air.CITY').alias('CITY'),
        col('total_departure_delay'),
        col('total_arrival_delay'),
        col('total_delay')
    )
    
    # Raggruppa per città e somma i ritardi
    city_delays = total_delays.groupBy('CITY').agg(
        sum('total_departure_delay').alias('total_departure_delay'),
        sum('total_arrival_delay').alias('total_arrival_delay'),
        sum('total_delay').alias('total_delay')
    )
    
    # Ordina per ritardo totale in ordine decrescente
    sorted_city_delays = city_delays.orderBy(col('total_delay').desc())
    
    return sorted_city_delays


# In[13]:


def airlines_with_most_delays(flights_df: DataFrame, airlines_df: DataFrame) -> DataFrame:
    # Calcola i ritardi totali in partenza per ciascuna compagnia aerea
    departure_delays = flights_df.groupBy('AIRLINE').agg(
        sum('DEPARTURE_DELAY').alias('total_departure_delay')
    )
    
    # Calcola i ritardi totali in arrivo per ciascuna compagnia aerea
    arrival_delays = flights_df.groupBy('AIRLINE').agg(
        sum('ARRIVAL_DELAY').alias('total_arrival_delay')
    )
    
    # Unisce i ritardi di partenza e arrivo per ciascuna compagnia aerea
    total_delays = departure_delays.join(
        arrival_delays,
        on='AIRLINE',
        how='outer'
    ).select(
        col('AIRLINE'),
        col('total_departure_delay'),
        col('total_arrival_delay')
    )
    
    # Somma i ritardi di partenza e arrivo
    total_delays = total_delays.withColumn(
        'total_delay',
        col('total_departure_delay') + col('total_arrival_delay')
    ).na.fill(0)  # Riempi i valori nulli con 0
    
    # Unisce con il DataFrame delle compagnie aeree per ottenere i nomi completi
    total_delays = total_delays.join(
        airlines_df,
        total_delays.AIRLINE == airlines_df.IATA_CODE,
        how='left'
    ).select(
        col('AIRLINE'),
        col('AIRLINE').alias('AIRLINE_CODE'),
        col('total_departure_delay'),
        col('total_arrival_delay'),
        col('total_delay'),
        col('AIRLINE').alias('AIRLINE_NAME')
    )
    
    # Ordina per ritardo totale in ordine decrescente
    sorted_total_delays = total_delays.orderBy(col('total_delay').desc())
    
    return sorted_total_delays


# In[ ]:





# In[14]:


def heatmap_delay(flights_df: DataFrame, airports_df: DataFrame, airlines_df:DataFrame ):
    
    # Rinomina le colonne duplicate per evitare conflitti durante l'unione
    airports_df = airports_df.withColumnRenamed("AIRLINE", "AIRLINE_AIRPORTS")
    airlines_df = airlines_df.withColumnRenamed("AIRLINE", "AIRLINE_NAME")
    
    # Unione delle tabelle sulla colonna IATACODE
    joined_df = flights_df.join(airports_df, flights_df.ORIGIN_AIRPORT == airports_df.IATA_CODE, "inner")                         .join(airlines_df, flights_df.AIRLINE == airlines_df.IATA_CODE, "inner")

    # Calcolare la media dei ritardi di partenza per città e compagnia aerea
    average_delays = joined_df.groupBy("CITY", "AIRLINE_NAME")                             .agg(mean("DEPARTURE_DELAY").alias("Average Departure Delay"))                            #.orderBy("Average Departure Delay", ascending=False)

    # Convertire il risultato in un Pandas DataFrame per la visualizzazione con Seaborn
    pandas_df = average_delays.toPandas()

    # Pivot della tabella per creare la matrice per la heatmap
    heatmap_data = pandas_df.pivot(index="CITY", columns="AIRLINE_NAME", values="Average Departure Delay")

    # Creare la heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data, cmap=sns.color_palette("flare", as_cmap=True), annot=False, vmax=30, vmin=0)

    # Aggiungere etichette, titolo e personalizzazione
    plt.title('Delays: impact of the origin airport')
    plt.xlabel('Airlines')
    plt.ylabel('Cities')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0) 
    #plt.ylim(0, 230)
    
    
    # Aumentare la spaziatura per le etichette degli assi
    plt.tight_layout(rect=[0, 0, 1, 1])
    # Mostrare il grafico
    plt.show()


# In[15]:


def heatmap_delay_dinamic(flights_df: DataFrame, airports_df: DataFrame, airlines_df:DataFrame ):
    
    # Rinomina le colonne duplicate per evitare conflitti durante l'unione
    airports_df = airports_df.withColumnRenamed("AIRLINE", "AIRLINE_AIRPORTS")
    airlines_df = airlines_df.withColumnRenamed("AIRLINE", "AIRLINE_NAME")
    
    # Unione delle tabelle sulla colonna IATACODE
    joined_df = flights_df.join(airports_df, flights_df.ORIGIN_AIRPORT == airports_df.IATA_CODE, "inner")                         .join(airlines_df, flights_df.AIRLINE == airlines_df.IATA_CODE, "inner")

    # Calcolare la media dei ritardi di partenza per città e compagnia aerea
    average_delays = joined_df.groupBy("CITY", "AIRLINE_NAME")                             .agg(mean("DEPARTURE_DELAY").alias("Average Departure Delay"))                            #.orderBy("Average Departure Delay", ascending=False)

    # Convertire il risultato in un Pandas DataFrame per la visualizzazione con Seaborn
    pandas_df = average_delays.toPandas()

    # Creare la heatmap interattiva con plotly
    fig = px.imshow(pandas_df.pivot(index="CITY", columns="AIRLINE_NAME", values="Average Departure Delay"),
                    labels=dict(x="Airlines", y="Cities", color="Average Departure Delay"),
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


# In[16]:


def plot_cities_with_most_delays(flights_df: DataFrame, airports_df: DataFrame, top_n: int = 10):
    # Calcola le città con più ritardi
    delays_df = cities_with_most_delays(flights_df, airports_df)
    
    # Converti il DataFrame in Pandas per Plotly
    delays_pd = delays_df.toPandas()
    
    # Prendi le prime top_n città con più ritardi
    top_delays_pd = delays_pd.head(top_n)
    
    # Creazione del grafico a barre impilate con Plotly Express
    top_delays_pd = top_delays_pd.melt(id_vars=['CITY'], value_vars=['total_departure_delay', 'total_arrival_delay'],
                                       var_name='Delay_Type', value_name='Total_Delay')
    
    fig = px.bar(top_delays_pd, x='CITY', y='Total_Delay', color='Delay_Type', 
                 labels={'CITY': 'City', 'Total_Delay': 'Total Delay', 'Delay_Type': 'Delay Type'},
                 title=f'Top {top_n} Cities with Most Delays')
    
    # Mostra il grafico
    fig.show()


# In[17]:


def plot_airlines_with_most_delays(flights_df: DataFrame, airlines_df: DataFrame, top_n: int = 10):
    # Calcola i ritardi per ciascuna compagnia aerea
    delays_df = airlines_with_most_delays(flights_df, airlines_df)
    
    # Converti il DataFrame in Pandas per Plotly
    delays_pd = delays_df.toPandas()
    
    # Prendi le prime top_n compagnie aeree con più ritardi
    top_delays_pd = delays_pd.head(top_n)
    
    # Creazione del grafico a barre impilate con Plotly Express
    top_delays_pd = top_delays_pd.melt(id_vars=['AIRLINE_NAME'], value_vars=['total_departure_delay', 'total_arrival_delay'],
                                       var_name='Delay_Type', value_name='Total_Delay')
    
    fig = px.bar(top_delays_pd, x='AIRLINE_NAME', y='Total_Delay', color='Delay_Type', 
                 labels={'AIRLINE_NAME': 'Airline', 'Total_Delay': 'Total Delay', 'Delay_Type': 'Delay Type'},
                 title=f'Top {top_n} Airlines with Most Delays')
    
    # Mostra il grafico
    fig.show()


# In[18]:


#################################################################################


# In[19]:


def plot_cancellation_reasons(cancelled_flights_df: DataFrame):
    
    # Raggruppa e conta i voli cancellati per motivo della cancellazione e li ordina in modo decrescente 
    cancellation_reasons_df = cancelled_flights_df.groupBy('CANCELLATION_REASON')         .agg(count('*').alias('count'))         .orderBy('count', ascending=False)         .toPandas()
    
    # Cause cancellazione volo
    cancellation_reason_map = {
    'A': 'Airline',
    'B': 'Weather',
    'C': 'Air Traffic Control',
    'D': 'Security'
    }
    
    # Sostituire i codici con i significati parlanti
    cancellation_reasons_df['CANCELLATION_REASON'] = cancellation_reasons_df['CANCELLATION_REASON'].map(cancellation_reason_map)
    
    plt.figure(figsize=(10, 6))
    
    #Creazione del grafico a barre
    bars = plt.bar(cancellation_reasons_df['CANCELLATION_REASON'], cancellation_reasons_df['count'], color='royalblue')
    
    #Etichette
    plt.xlabel('Cancellation Reason')
    plt.ylabel('Number of Cancellations')
    
    #Aggiunta titolo
    plt.title('Main Reasons for Flight Cancellations')
    
    # Aggiungere i numeri delle cancellazioni sopra le barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 10, int(yval), ha='center', va='bottom', fontsize=11)
    
    #Mostrare il grafico
    plt.show()


# In[20]:


def delays_per_airport(flights_df: DataFrame, airports_df: DataFrame) -> DataFrame:
    # Calcola i ritardi totali in partenza per ciascun aeroporto
    departure_delays = flights_df.groupBy('ORIGIN_AIRPORT').agg(
        
        sum('DEPARTURE_DELAY').alias('total_departure_delay')
    )
    
    # Calcola i ritardi totali in arrivo per ciascun aeroporto
    arrival_delays = flights_df.groupBy('DESTINATION_AIRPORT').agg(
        sum('ARRIVAL_DELAY').alias('total_arrival_delay')
    )
    
    # Unisce i ritardi di partenza e arrivo per ciascun aeroporto
    total_delays = departure_delays.join(
        arrival_delays,
        departure_delays.ORIGIN_AIRPORT == arrival_delays.DESTINATION_AIRPORT,
        how='outer'
    ).select(
        col('ORIGIN_AIRPORT').alias('AIRPORT'),
        col('total_departure_delay'),
        col('total_arrival_delay')
    ).na.fill(0)
    
    # Somma i ritardi di partenza e arrivo
    total_delays = total_delays.withColumn(
        'total_delay',
        col('total_departure_delay') + col('total_arrival_delay')
    )
    
    # Unisce con il DataFrame degli aeroporti per ottenere i nomi completi
    total_delays = total_delays.join(
        airports_df,
        total_delays.AIRPORT == airports_df.IATA_CODE,
        how='left'
    ).select(
        col('AIRPORT'),
        col('total_departure_delay'),
        col('total_arrival_delay'),
        col('total_delay'),
        col('AIRPORT').alias('AIRPORT_NAME')
    )
    
    # Ordina per ritardo totale in ordine decrescente
    sorted_total_delays = total_delays.orderBy(col('total_delay').desc())
    
    return sorted_total_delays


# In[21]:


def plot_cancellations_by_month(cancelled_flights_df: DataFrame):
    
    #Raggruppa ed ordina per mese (January = 1, ..., December = 12)
    cancellations_count = cancelled_flights_df.groupBy('MONTH').count()
    cancellations_count = cancellations_count.sort('MONTH')
    cancellations_pd = cancellations_count.toPandas()
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Creazione grafico a barre
    plt.bar(cancellations_pd['MONTH'], cancellations_pd['count'], color='lightcoral')
    
    # Titolo
    plt.title('Number of cancelled flights per month')
    
    #Etichette
    plt.xlabel('Month')
    plt.ylabel('Number of cancelled flights')
    
    #Mesi
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(ticks=range(0, 12), labels=months)
    
    #Riga che parte dall'asse delle ordinate
    plt.grid(axis='y')
    plt.show()


# In[22]:


def plot_cancellations_by_day_of_week(cancelled_flights_df: DataFrame):
    
    cancellations_count = cancelled_flights_df.groupBy('DAY_OF_WEEK').count()
    
    # Ordina i risultati per il giorno della settimana (Monday = 1, ..., Sunday = 7)
    cancellations_count = cancellations_count.sort('DAY_OF_WEEK')

    cancellations_pd = cancellations_count.toPandas()
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    #Creazione del grafico a barre
    plt.bar(cancellations_pd['DAY_OF_WEEK'], cancellations_pd['count'], color='skyblue')
    
    #Titolo
    plt.title('Number of cancelled flights per day of the week')
    
    #Etichette
    plt.xlabel('Day of the week')
    plt.ylabel('Number of cancelled flights')
    
    #Giorni della settimana
    plt.xticks(range(1, 8), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    #Riga che parte dall'asse delle ordinate
    plt.grid(axis='y')
    plt.show()


# In[23]:


def plot_cancellations_per_airport(flights_df: DataFrame, airports_df: DataFrame, top_n: int = 10):
    # Calcola le cancellazioni per ciascun aeroporto
    cancellations_df = cancellations_per_airport(flights_df, airports_df)
    
    # Converti il DataFrame in Pandas per Plotly
    cancellations_pd = cancellations_df.toPandas()
    
    # Prendi i primi top_n aeroporti con più cancellazioni
    top_cancellations_pd = cancellations_pd.head(top_n)
    
    # Creazione del grafico a barre con Plotly Express
    fig = px.bar(top_cancellations_pd, x='AIRPORT_NAME', y='total_cancellations',
                 labels={'AIRPORT_NAME': 'Airport', 'total_cancellations': 'Total Cancellations'},
                 title=f'Top {top_n} Airports with Most Cancellations')
    
    # Mostra il grafico
    fig.show()

def delays_during_the_day():
    pass

# In[24]:


def main():
    # Creazione della sessione Spark
    spark = SparkSession.builder.appName("Flight Delays and Cancellations Analysis").getOrCreate()

    columns_to_drop_init = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR', 
                            'DAY','DATE', 'AIR_SYSTEM_DELAY',
                            'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                            'WEATHER_DELAY', 'DIVERTED', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                            'AIR_TIME', 'HOUR', 'MINUTE']
    # Percorso ai file CSV
    file_path_flights = "C:\\Users\\utente\\Desktop\\MasterDS_TVG\\Big Data\\progetto_finale_big_data\\flights.csv"
    file_path_airport = "C:\\Users\\utente\\Desktop\\MasterDS_TVG\\Big Data\\progetto_finale_big_data\\airports.csv"
    file_path_airline = "C:\\Users\\utente\\Desktop\\MasterDS_TVG\\Big Data\\progetto_finale_big_data\\airlines.csv"
    # Caricamento dei dataset
    flights_df = spark.read.csv(file_path_flights, header=True, inferSchema=True)
    airports_df = spark.read.csv(file_path_airport, header=True, inferSchema=True)
    airlines_df = spark.read.csv(file_path_airline, header=True, inferSchema=True)
    # Unisce le colonne che rappresentano anno, mese, giorno e orario di partenza programmata in una sola colonna
    flights_df = conc_date(flights_df)
    # Trasforma l'orario del dataset in un formato più leggibile (HH:MM)
    flights_df = trasform_date(flights_df, "DEPARTURE_TIME")
    flights_df = trasform_date(flights_df, "SCHEDULED_ARRIVAL")
    flights_df = trasform_date(flights_df, "ARRIVAL_TIME")
    # Elimina le colonne che non servono
    flights_df = drop_column(flights_df, columns_to_drop_init)
    # Divide in due i dataset
    voli_cancellati, voli_in_ritardo=divide_dataset(flights_df)

    columns_to_drop_cancellati = ['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ELAPSED_TIME',
                                'ARRIVAL_TIME', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']

    columns_to_drop_ritardo = ['CANCELLED','CANCELLATION_REASON']

    voli_cancellati = drop_column(voli_cancellati, columns_to_drop_cancellati)
    voli_in_ritardo = drop_column(voli_in_ritardo, columns_to_drop_ritardo)

    # eliminati 15187 su 5729195 equivalente al 0,27% si può optare pure un imputazione
    #voli_in_ritardo = delete_rows_null(voli_in_ritardo, ["ELAPSED_TIME"])
    
    #df=distribuzione_valori_nulli(voli_in_ritardo, "ARRIVAL_TIME")
     
    #funzione che mostra in un grafico le città con maggiore ritardo in ingresso e uscita
    most_delay(voli_in_ritardo, airports_df)
    
    #funzione che mostra il totale dei voli partiti e arrivati in ritardo per città
    cities_with_most_delays(voli_in_ritardo, airports_df)
    
    #funzione che disegna una heatmap fra aeroporti e compagnie,
    #i valori rappresentano invece la media dei minuti di ritardo
    heatmap_delay(voli_in_ritardo, airports_df, airlines_df)
    
    #La stessa cosa della funzione sopra, questa invece è più dinamica
    heatmap_delay_dinamic(voli_in_ritardo, airports_df, airlines_df)
    
    #Funzione che disegna il grafico del totale dei voli arrivati in arrivo ed in ritardo per ciascuna città
    plot_cities_with_most_delays(voli_in_ritardo, airports_df, 10)
    
    #funzione che mostra le principali motivazioni delle cancellazioni dei voli
    plot_cancellation_reasons(voli_cancellati)
    
    #Funzione che mostra il numero di cancellazioni suddivise per mesi
    plot_cancellations_by_month(voli_cancellati)
    
    #Funzione che mostra il numero di cancellazioni suddivise per giorno della settimana
    plot_cancellations_by_day_of_week(voli_cancellati)
    
    #Funzione che mostra le compagnie aeree con più ritardi
    plot_airlines_with_most_delays(flights_df, 10)
    
    # Funzione che calcola e mostra le cancellazioni per aeroporto
    plot_cancellations_per_airport(flights_df, airports_df, 10)
    
    # Funzione che calcola e mostra i ritardi per aeroporto
    plot_delays_per_airport(flights_df, airports_df, 10)
      
    print(0)


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:


main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




