from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, mean, count, hour, avg, lit, \
                                     sum as _sum, concat_ws, coalesce

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import FuncFormatter

class DelayedFlights:
    def __init__(self, delayed_flights: DataFrame, airports_df: DataFrame, airline_df: DataFrame):
        self.delayed_flights = delayed_flights
        self.airports_df = airports_df 
        self.airline_df = airline_df

    def most_delay(self):
        # Unione delle tabelle sulla colonna IATACODE
        joined_df = self.delayed_flights.join(
            self.airports_df, self.delayed_flights.ORIGIN_AIRPORT == self.airports_df.IATA_CODE, "inner")

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
            top_cities['CITY'], top_cities['Average Departure Delay'], color='b', label='Ritardo alla partenza')
        bars2 = plt.barh(top_cities['CITY'], top_cities['Average Arrival Delay'],
                        color='r', label="Ritardo all'arrivo", left=top_cities['Average Departure Delay'])

        # Aggiungere i numeri sul grafico
        for bar1, bar2 in zip(bars1, bars2):
            # Numeri al centro della barra blu
            plt.text(bar1.get_width() / 2, bar1.get_y() + bar1.get_height() / 2,
                    f'{bar1.get_width():.1f}', va='center', ha='center', color='white', fontsize=8, weight='bold')
            # Numeri alla fine della barra rossa
            plt.text(bar2.get_width() + bar1.get_width() + 0.2, bar2.get_y() + bar2.get_height() / 2,
                    f'{bar2.get_width():.1f}', va='center', ha='left', color='red', fontsize=8)

        # Aggiungere etichette, titolo e legenda
        plt.xlabel('Ritardo medio (minuti)')
        plt.ylabel('Città')
        plt.title('Ritardo medio di partenza e arrivo per città')
        plt.legend()

        # Mostrare il grafico
        plt.tight_layout()
        plt.show()

    def graph_city_airline_delay(self):

        # Rinomina le colonne duplicate per evitare conflitti durante l'unione
        airports_df = self.airports_df.withColumnRenamed("AIRLINE", "AIRLINE_AIRPORTS")
        airlines_df = self.airline_df.withColumnRenamed("AIRLINE", "AIRLINE_NAME")

        # Unione delle tabelle sulla colonna IATACODE
        joined_df = self.delayed_flights.join(airports_df, self.delayed_flights.ORIGIN_AIRPORT == airports_df.IATA_CODE, "inner") \
            .join(airlines_df, self.delayed_flights.AIRLINE == airlines_df.IATA_CODE, "inner")

        flights_count = joined_df.groupBy("CITY").agg(
            count("*").alias("NUM_FLIGHTS"))

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

    def origin_airport_pie(self):
        # Fare la join tra flights e airports per ottenere i nomi degli aeroporti
        flights_with_airport_names = self.delayed_flights.join(
            self.airports_df,
            self.delayed_flights.ORIGIN_AIRPORT == self.airports_df.IATA_CODE,
            "inner"
        ).select(
            self.delayed_flights["*"],
            self.airports_df["AIRPORT"].alias("ORIGIN_AIRPORT_NAME")
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
        
        # Convertire il DataFrame PySpark in un Pandas DataFrame
        flights_pandas = flights_percentage_first.toPandas()

        # Creare il grafico a torta
        fig = px.pie(flights_pandas, values="NUM_FLIGHTS", names="ORIGIN_AIRPORT_NAME",
                    title="Percentuale di voli per aeroporto di origine")

        # Mostrare il grafico
        fig.show()

    def route_most_delay(self):

        df_flights = self.delayed_flights.withColumn("ROUTE", concat_ws(
            "-", self.delayed_flights.ORIGIN_AIRPORT, self.delayed_flights.DESTINATION_AIRPORT))

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
                        "TOTAL_AVG_DELAY": "Ritardo medio totale (minuti)", "ROUTE": "Rotte"},
                    title="Top 10 rotte con il più grande ritardo medio totale")

        # Mostrare il grafico
        fig.show()

    def most_delay_airport_most_flight(self):
        # Calcolare il numero di voli per ogni aeroporto di partenza
        flights_count = self.delayed_flights.groupBy("ORIGIN_AIRPORT").agg(
            count("*").alias("NUM_FLIGHTS"))

        # Calcolare il ritardo medio di partenza per ogni aeroporto
        avg_departure_delay = self.delayed_flights.groupBy("ORIGIN_AIRPORT").agg(
            mean("DEPARTURE_DELAY").alias("AVG_DEPARTURE_DELAY"))

        flights_analysis = flights_count.join(
            avg_departure_delay, on="ORIGIN_AIRPORT")

        flights_analysis = flights_analysis.orderBy(
            "AVG_DEPARTURE_DELAY", ascending=False)

        # Convertire il DataFrame PySpark in un Pandas DataFrame
        flights_analysis_pandas = flights_analysis.toPandas()

        # Creare il grafico a barre
        fig = px.scatter(flights_analysis_pandas, x="NUM_FLIGHTS", y="AVG_DEPARTURE_DELAY",
                        labels={"NUM_FLIGHTS": "Numeri di voli",
                                "AVG_DEPARTURE_DELAY": "Ritardo medio di partenza (minuti)"},
                        title="Relazione fra numero di voli e ritardo medio di partenza per aereoporto")

        # Mostrare il grafico
        fig.show()

    def __cities_with_most_delays(self):
        '''
        Funzione che restituisce le città ordinate in base al numero
        di voli partiti o arrivati in ritardo
        '''
        # Calcola i ritardi totali in partenza per ogni aeroporto
        departure_delays = self.delayed_flights.groupBy('ORIGIN_AIRPORT').agg(
            _sum('DEPARTURE_DELAY').alias('total_departure_delay')
        )
        
        # Calcola i ritardi totali in arrivo per ogni aeroporto
        arrival_delays = self.delayed_flights.groupBy('DESTINATION_AIRPORT').agg(
            _sum('ARRIVAL_DELAY').alias('total_arrival_delay')
        )
        
        # Assegna alias ai DataFrame
        departure_delays = departure_delays.alias('dep')
        arrival_delays = arrival_delays.alias('arr')
        airports_df = self.airports_df.alias('air')
        
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
            _sum('total_departure_delay').alias('total_departure_delay'),
            _sum('total_arrival_delay').alias('total_arrival_delay'),
            _sum('total_delay').alias('total_delay')
        )
        
        # Ordina per ritardo totale in ordine decrescente
        sorted_city_delays = city_delays.orderBy(col('total_delay').desc())
        
        return sorted_city_delays
    
    def plot_cities_with_most_delays(self, top_n=10):
        '''
        Funzione che, partendo dalla funzione cities_with_most_delays(),
        restituisce le 10 città con più voli in ritardo
        '''
        # Calcola le città con più ritardi
        delays_df = self.__cities_with_most_delays()
        
        # Converti il DataFrame in Pandas per Plotly
        delays_pd = delays_df.toPandas()
        
        # Prendi le prime top_n città con più ritardi
        top_delays_pd = delays_pd.head(top_n)
        
        # Creazione del grafico a barre impilate con Plotly Express
        top_delays_pd = top_delays_pd.melt(id_vars=['CITY'], value_vars=['total_departure_delay', 'total_arrival_delay'],
                                        var_name='Delay_Type', value_name='Total_Delay')
        
        # Rinomina i tipi di ritardo per la legenda e il grafico
        top_delays_pd['Delay_Type'] = top_delays_pd['Delay_Type'].replace({
            'total_departure_delay': 'Partenza',
            'total_arrival_delay': 'Arrivo'
        })
        
        fig = px.bar(top_delays_pd, x='CITY', y='Total_Delay', color='Delay_Type', 
                    labels={'CITY': 'Città', 'Total_Delay': 'Totale ritardo (in minuti)', 'Delay_Type': 'Tipologia ritardo'},
                    title=f'Top {top_n} Città con più minuti di ritardo accumulati',
                    category_orders={'Delay_Type': ['Total Departure Delay', 'Total Arrival Delay']},
                    text_auto=True)  # Aggiunge i numeri sulle barre
        
        # Personalizzazione della legenda e delle etichette
        fig.update_layout(
            legend_title_text='Totale minuti di ritardo',  
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(
                    family="Arial",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2
            )
        )
        
        # Imposta l'angolo del testo a 0 gradi per renderli orizzontali
        fig.update_traces(textangle=0)
        
        # Mostra il grafico
        fig.show()

    def __time_slots(self):
        '''
        Funzione che divide i ritardi in fasce orarie
        '''
        # Aggiunge una colonna 'HOUR' con l'ora di partenza programmata
        flights_df = self.delayed_flights.withColumn("HOUR", hour(col("SCHEDULED_DEPARTURE")))
    
        # Filtra i voli con ritardo alla partenza
        delayed_flights_df = flights_df.filter(col("DEPARTURE_DELAY") > 0)
        
        # Conta i ritardi per ogni ora e ordina per numero totale di ritardi in modo decrescente
        hourly_delays_count = delayed_flights_df.groupBy("HOUR").agg(
            count("DEPARTURE_DELAY").alias("Total Delays")
        ).orderBy("Total Delays", ascending=False)
        
        return hourly_delays_count
    
    def avg_delay(self):
        '''
        Funzione che restituisce un grafico a barre orizzontali riportando la media
        dei minuti di ritardo di ciascuna compagnia aerea
        '''
        # Mappa dei codici IATA alle compagnie aeree complete
        airline_name_mapping = {
            "UA": "United Air Lines Inc.",
            "AA": "American Airlines Inc.",
            "US": "US Airways Inc.",
            "F9": "Frontier Airlines Inc.",
            "B6": "JetBlue Airways",
            "OO": "Skywest Airlines Inc.",
            "AS": "Alaska Airlines Inc.",
            "NK": "Spirit Air Lines",
            "WN": "Southwest Airlines Co.",
            "DL": "Delta Air Lines Inc.",
            "EV": "Atlantic Southeast Airlines",
            "HA": "Hawaiian Airlines Inc.",
            "MQ": "American Eagle Airlines Inc.",
            "VX": "Virgin America"
        }

        # Filtra i voli con ritardi di partenza o arrivo
        voli_in_ritardo_df = self.delayed_flights.filter((col("DEPARTURE_DELAY") > 0) | (col("ARRIVAL_DELAY") > 0))

        # Calcola il numero di voli in ritardo per compagnia aerea
        compagnie_voli_in_ritardo = voli_in_ritardo_df.groupBy("AIRLINE").agg(count("ARRIVAL_DELAY").alias("NUM_DELAYED_FLIGHTS"))

        # Calcola la media dei minuti di ritardo di partenza e arrivo per compagnia aerea
        compagnie_media_ritardo = voli_in_ritardo_df.groupBy("AIRLINE").agg(
            avg("DEPARTURE_DELAY").alias("AVG_DEPARTURE_DELAY"),
            avg("ARRIVAL_DELAY").alias("AVG_ARRIVAL_DELAY")
        )

        # Aggiungi una colonna con la media dei ritardi totali
        compagnie_media_ritardo = compagnie_media_ritardo.withColumn(
            "AVG_TOTAL_DELAY", 
            (col("AVG_DEPARTURE_DELAY") + col("AVG_ARRIVAL_DELAY")) / 2
        )

        # Converti i DataFrame in Pandas
        compagnie_voli_in_ritardo_pd = compagnie_voli_in_ritardo.toPandas()
        compagnie_media_ritardo_pd = compagnie_media_ritardo.toPandas()

        # Arrotonda le medie a zero cifre decimali
        compagnie_media_ritardo_pd['AVG_DEPARTURE_DELAY'] = compagnie_media_ritardo_pd['AVG_DEPARTURE_DELAY'].round(0).astype(int)
        compagnie_media_ritardo_pd['AVG_ARRIVAL_DELAY'] = compagnie_media_ritardo_pd['AVG_ARRIVAL_DELAY'].round(0).astype(int)
        compagnie_media_ritardo_pd['AVG_TOTAL_DELAY'] = compagnie_media_ritardo_pd['AVG_TOTAL_DELAY'].round(0).astype(int)

        # Espandi i nomi delle compagnie aeree
        compagnie_voli_in_ritardo_pd['AIRLINE'] = compagnie_voli_in_ritardo_pd['AIRLINE'].map(airline_name_mapping).fillna(compagnie_voli_in_ritardo_pd['AIRLINE'])
        compagnie_media_ritardo_pd['AIRLINE'] = compagnie_media_ritardo_pd['AIRLINE'].map(airline_name_mapping).fillna(compagnie_media_ritardo_pd['AIRLINE'])

        # Visualizza le compagnie aeree con la media più alta di minuti di ritardo
        plt.figure(figsize=(12, 8))
        plot2 = sns.barplot(data=compagnie_media_ritardo_pd.sort_values(by="AVG_TOTAL_DELAY", ascending=False), 
                        x="AVG_TOTAL_DELAY", y="AIRLINE", palette="viridis", alpha=0.8)
        plot2.set_title("Media minuti di ritardo per compagnia aerea")
        plot2.set_xlabel("Media minuti di ritardo")
        plot2.set_ylabel("Compagnia aerea")
        
        # Aggiungi le annotazioni sui grafici
        for p in plot2.patches:
            width = int(p.get_width())  # Arrotonda il valore a un numero intero
            plot2.annotate(f'{width}',  # Visualizza l'intero senza cifre decimali
                        (width, p.get_y() + p.get_height() / 2),
                        ha='center', va='center',
                        xytext=(10, 0),
                        textcoords='offset points')
        
        plt.show()

    def __avg_calculation(self):
        '''
        Funzione che calcola la media dei minuti di ritardo
        '''
        # Aggiunge una colonna 'HOUR_OF_DAY' con l'ora di partenza programmata
        flights_df = self.delayed_flights.withColumn("HOUR_OF_DAY", hour(col("SCHEDULED_DEPARTURE")))
        
        # Calcola la media dei ritardi alla partenza per ogni ora del giorno
        avg_departure_delay = flights_df.groupBy("HOUR_OF_DAY").agg(
            mean("DEPARTURE_DELAY").alias("Average Departure Delay")
        ).orderBy("HOUR_OF_DAY")
        
        # Calcola la media dei ritardi all'arrivo per ogni ora del giorno
        avg_arrival_delay = flights_df.groupBy("HOUR_OF_DAY").agg(
            mean("ARRIVAL_DELAY").alias("Average Arrival Delay")
        ).orderBy("HOUR_OF_DAY")
        
        # Unisce i risultati delle medie dei ritardi alla partenza e all'arrivo
        avg_delays = avg_departure_delay.join(avg_arrival_delay, on="HOUR_OF_DAY", how="outer")
        
        return avg_delays

    def delay_analysis(self):
        '''
        Funzione che partendo da time_slots() e avg_calculation()
        calcola la media dei minuti di ritardo per ciascuna fascia oraria
        '''
        # Ottiene il conteggio dei ritardi per fascia oraria
        hourly_delays_count = self.__time_slots()
        
        # Ottiene la media dei ritardi per fascia oraria
        avg_delays = self.__avg_calculation()
        
        # Unisce i due DataFrame e seleziona le colonne di interesse, rinominando per leggibilità
        result_df = hourly_delays_count.join(avg_delays, hourly_delays_count.HOUR == avg_delays.HOUR_OF_DAY, how="outer").select(
            col("HOUR").alias("Hour"),
            "Total Delays",
            "Average Departure Delay",
            "Average Arrival Delay"
        ).orderBy("Hour")
        
        # Mostra i risultati
        #result_df.show()

        # Converte il risultato in un DataFrame Pandas
        result_pd_df = result_df.toPandas()

        # Funzione per determinare la fascia oraria
        def get_time_slot(hour):
            if 6 <= hour <= 11:
                return 'Mattina (6 - 12)'
            elif 12 <= hour <= 18:
                return 'Pomeriggio (12 - 19)'
            elif 19 <= hour <= 24:
                return 'Sera (19 - 1)'
            else:
                return 'Notte (1 - 6)'

        # Aggiunge una colonna 'Time Slot' al DataFrame Pandas
        result_pd_df['Time Slot'] = result_pd_df['Hour'].apply(get_time_slot)
        
        
        # Calcola il numero totale di ritardi per fascia oraria
        total_delays_per_slot = result_pd_df.groupby('Time Slot')['Total Delays'].sum()

        # Crea un grafico a barre per il numero totale di ritardi per fascia oraria
        plt.figure(figsize=(10, 6))
        sns.barplot(x=total_delays_per_slot.index, y=total_delays_per_slot.values, palette="Set2")
        plt.title('Totale ritardi per fascia oraria', fontsize=16)
        plt.xlabel('Fascia oraria', fontsize=14)
        plt.ylabel('Ritardi  totali', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Aggiunge i numeri precisi sulle colonne
        for i, v in enumerate(total_delays_per_slot.values):
            plt.text(i, v + 5, f'{v:.0f}', ha='center', fontsize=12)

        # Formattazione dell'asse delle ordinate per numeri grandi
        def format_ordinate(x, _):
            if x >= 1000:
                return f'{int(x/1000)}k'
            return f'{int(x)}'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ordinate))

        plt.tight_layout()
        plt.show()


        # Calcola la media dei ritardi per fascia oraria
        avg_delay_per_slot = result_pd_df.groupby('Time Slot').agg({
            'Average Departure Delay': 'mean',
            'Average Arrival Delay': 'mean'
        })

        # Crea un grafico a barre per la media dei ritardi per fascia oraria
        ax = avg_delay_per_slot.plot(kind='bar', title='Ritardo medio per fasce orarie', figsize=(10, 6), colormap='Set2')
        plt.title('Ritardo medio per fascia oraria', fontsize=16)
        plt.xlabel('Fascia oraria', fontsize=14)
        plt.ylabel('Ritardo medio (in minuti)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Aggiunge i numeri precisi sulle colonne
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()
        
