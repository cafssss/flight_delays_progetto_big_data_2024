from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, mean, count, when, lpad, concat, lit, \
    to_timestamp, month, year, sum as _sum, concat_ws, udf, \
    monotonically_increasing_id

import matplotlib.pyplot as plt
import seaborn as sns



class CancelledFlights:
    def __init__(self, cancelled_flights: DataFrame):
        self.cancelled_flights = cancelled_flights

    def plot_cancellation_reasons(self):
        '''
        Funzione che restituisce il grafico a barre delle motivazioni relative
        alle cancellazioni dei voli
        '''
        # Raggruppa e conta i voli cancellati per motivo della cancellazione e li ordina in modo decrescente
        cancellation_reasons_df = self.cancelled_flights.groupBy('CANCELLATION_REASON').agg(
            count('*').alias('count')).orderBy('count', ascending=False).toPandas()

        # Cause cancellazione volo
        cancellation_reason_map = {
            'A': 'Compagnia aerea',
            'B': 'Meteo',
            'C': 'Traffico aereo',
            'D': 'Sicurezza'
        }

        # Sostituire i codici con i significati parlanti
        cancellation_reasons_df['CANCELLATION_REASON'] = cancellation_reasons_df['CANCELLATION_REASON'].map(
            cancellation_reason_map)

        plt.figure(figsize=(10, 6))

        # Creazione del grafico a barre
        bars = plt.bar(cancellation_reasons_df['CANCELLATION_REASON'],
                       cancellation_reasons_df['count'], color='royalblue')

        # Etichette
        plt.xlabel('Motivo cancellazione')
        plt.ylabel('Numero di cancellazioni')

        # Aggiunta titolo
        plt.title('Motivazioni voli cancellati')

        # Aggiungere i numeri delle cancellazioni sopra le barre
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 10,
                     int(yval), ha='center', va='bottom', fontsize=11)

        # Mostrare il grafico
        plt.show()

    def plot_cancellation_reasons_pie(self):
        '''
        Funzione che restituisce il grafico a torta delle motivazioni relative
        alle cancellazioni dei voli
        '''
        # Raggruppa e conta i voli cancellati per motivo della cancellazione e li ordina in modo decrescente
        cancellation_reasons_df = self.cancelled_flights.groupBy('CANCELLATION_REASON').agg(
            count('*').alias('count')).orderBy('count', ascending=False).toPandas()

        # Cause cancellazione volo
        cancellation_reason_map = {
            'A': 'Compagnia aerea',
            'B': 'Meteo',
            'C': 'Traffico aereo',
            'D': 'Sicurezza'
        }

        # Sostituire i codici con i significati parlanti
        cancellation_reasons_df['CANCELLATION_REASON'] = cancellation_reasons_df['CANCELLATION_REASON'].map(
            cancellation_reason_map)

        # Calcolare le percentuali
        total_cancellations = cancellation_reasons_df['count'].sum()
        cancellation_reasons_df['percentage'] = (
            cancellation_reasons_df['count'] / total_cancellations) * 100

        # Creazione del grafico a torta
        plt.figure(figsize=(8, 8))
        plt.pie(cancellation_reasons_df['percentage'],
                labels=cancellation_reasons_df['CANCELLATION_REASON'],
                autopct='%1.1f%%',
                startangle=140,
                colors=['#FF5733', '#33FF57', '#3357FF',
                        '#FF33A6'],  # Nuovi colori
                # Ingrandisce e inspessisce il testo
                textprops={'fontsize': 14, 'fontweight': 'bold'},
                # Imposta l'opacità dei segmenti
                wedgeprops=dict(edgecolor='black', linewidth=2, alpha=0.7))

        # Aggiunta titolo
        plt.title('Motivazioni voli cancellati')

        # Mostrare il grafico
        plt.show()

    def plot_cancellations_by_month(self):
        '''
        Funzione che restituisce il grafico a barre dei voli cancellati 
        suddivisi in base al mese
        '''
        #Raggruppa ed ordina per mese (January = 1, ..., December = 12)
        cancellations_count = self.cancelled_flights.groupBy('MONTH').count()
        cancellations_count = cancellations_count.sort('MONTH')
        cancellations_pd = cancellations_count.toPandas()
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Creazione grafico a barre
        plt.bar(cancellations_pd['MONTH'], cancellations_pd['count'], color='lightcoral')
        
        # Aggiungi numeri totali sopra ciascuna barra
        for index, value in enumerate(cancellations_pd['count']):
            plt.text(index, value, str(value), ha='center', va='bottom')
        
        # Titolo
        plt.title('Numero di cancellazioni per mese')
        
        #Etichette
        plt.xlabel('Mese')
        plt.ylabel('Numero voli cancellati')
        
        #Mesi
        months = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        plt.xticks(ticks=range(0, 12), labels=months)
        
        #Riga che parte dall'asse delle ordinate
        plt.grid(axis='y')
        plt.show()

    def plot_cancellations_by_day_of_week(self):
        '''
        Funzione che restituisce il grafico a barre dei voli cancellati 
        suddivisi in base al giorno della settimana
        '''
        cancellations_count = self.cancelled_flights.groupBy('DAY_OF_WEEK').count()
        
        # Ordina i risultati per il giorno della settimana (Monday = 1, ..., Sunday = 7)
        cancellations_count = cancellations_count.sort('DAY_OF_WEEK')

        cancellations_pd = cancellations_count.toPandas()
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        #Creazione del grafico a barre
        plt.bar(cancellations_pd['DAY_OF_WEEK'], cancellations_pd['count'], color='skyblue')
        
        # Aggiungi numeri totali sopra ciascuna barra
        for index, value in enumerate(cancellations_pd['count']):
            plt.text(index + 1, value, str(value), ha='center', va='bottom')
        
        #Titolo
        plt.title('Voli cancellati in base ai giorni della settimana')
        
        #Etichette
        plt.xlabel('Giorno della settimana')
        plt.ylabel('Numero di voli cancellati')
        
        #Giorni della settimana
        plt.xticks(range(1, 8), ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom'])
        
        #Riga che parte dall'asse delle ordinate
        plt.grid(axis='y')
        plt.show()

    def cancellation_per_airline(self):
        '''
        Funzione che restituisce il grafico a barre orizzontali
        delle cancellazioni suddivise per compagnia aerea
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

        # Conta le cancellazioni per compagnia aerea
        compagnie_cancellazioni = self.cancelled_flights.groupBy("AIRLINE").agg(count("CANCELLED").alias("NUM_CANCELLATIONS"))

        # Converti in Pandas DataFrame
        compagnie_cancellazioni_pd = compagnie_cancellazioni.toPandas()

        # Espandi i nomi delle compagnie aeree
        compagnie_cancellazioni_pd['AIRLINE'] = compagnie_cancellazioni_pd['AIRLINE'].map(airline_name_mapping).fillna(compagnie_cancellazioni_pd['AIRLINE'])
        
        # Crea un grafico delle cancellazioni per compagnia aerea
        plt.figure(figsize=(12, 8))
        plot = sns.barplot(data=compagnie_cancellazioni_pd.sort_values(by="NUM_CANCELLATIONS", ascending=False), 
                        x="NUM_CANCELLATIONS", y="AIRLINE", palette="Set1")  # Cambia palette qui
        plt.title("Numero di cancellazioni per compagnia aerea")
        plt.xlabel("Numero di cancellazioni")
        plt.ylabel("Compagnia aerea")
        
        # Aggiungi le annotazioni sui grafici
        for p in plot.patches:
            plot.annotate(format(p.get_width(), ',.0f'),  # Rimuovi i decimali
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='center', va='center',
                        xytext=(10, 0),
                        textcoords='offset points')
        
        plt.show()

    def cancellations_per_airport(self):
        '''
        Funzione che restituisce il grafico a barre delle cancellazioni suddivise per aeroporto
        '''

        # Conta le cancellazioni per aeroporto di partenza
        cancellations_per_origin_airport = self.cancelled_flights.groupBy("ORIGIN_AIRPORT").agg(
            count("CANCELLED").alias("NUM_CANCELLATIONS")
        )

        # Conta le cancellazioni per aeroporto di arrivo
        cancellations_per_destination_airport = self.cancelled_flights.groupBy("DESTINATION_AIRPORT").agg(
            count("CANCELLED").alias("NUM_CANCELLATIONS")
        )

        # Converti in Pandas DataFrame
        cancellations_per_origin_airport_pd = cancellations_per_origin_airport.toPandas()
        cancellations_per_destination_airport_pd = cancellations_per_destination_airport.toPandas()

        # Ordina e seleziona i primi 10 aeroporti per numero di cancellazioni (partenza)
        top_10_origin_airports = cancellations_per_origin_airport_pd.sort_values(by="NUM_CANCELLATIONS", ascending=False).head(10)

        # Visualizza le cancellazioni per aeroporto di partenza
        plt.figure(figsize=(15, 8))
        plot1 = sns.barplot(data=top_10_origin_airports, x="NUM_CANCELLATIONS", y="ORIGIN_AIRPORT", palette="Set2")  
        plt.title("Top 10 aeroporti per voli cancellati")
        plt.xlabel("Numero di cancellazioni")
        plt.ylabel("Aeroporto d'origine")
        
        # Aggiungi le annotazioni sui grafici
        for p in plot1.patches:
            plot1.annotate(format(p.get_width(), ',.0f'),  # Rimuovi i decimali
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='center', va='center',
                        xytext=(10, 0),
                        textcoords='offset points')
        
        plt.show()
