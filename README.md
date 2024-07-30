# Flight Delay and Cancellation Analysis

## Descrizione

Questo progetto analizza i ritardi e le cancellazioni dei voli utilizzando PySpark. Comprende il preprocessing dei dati, l'analisi esplorativa e la visualizzazione dei risultati.

## Struttura delle Cartelle

- `src/`: Contiene i file sorgente del progetto.

## Struttura del Codice

- `main.py`: Script principale che esegue l'analisi.
- `presentazione_finale.pptx`: Presentazione dell'analisi
- `src/preprocessing.py`: Contiene le funzioni per il preprocessing dei dati.
- `src/cancelled_flights.py`: Contiene le funzioni per l'analisi dei dati per i voli cancellati.
- `src/delayed_flights.py`: Contiene le funzioni per l'analisi dei dati per i voli in ritardo.
- `src/graphframes_flights.py`: Contiene le funzioni per l'analisi dei grafi con la libreria Graphframe.


## Requisiti

- Nella cartella è presente un file chiamato `requirements.txt`, contenente tutte le librerie necessarie
- Installare Spark e Hadoop

## Installazione

1. Clona il repository:
    ```bash
    git clone https://github.com/cafssss/flight_delays_progetto_big_data_2024.git
    ```

2. Installa le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```

3. Crea un file `.env` nella radice del progetto e configura i path ai dataset:
    ```ini
    # file .env
    PATH_FLIGHTS = path/to/data.csv
    PATH_AIRPORT = path/to/data.csv
    PATH_AIRLINE = path/to/data.csv
    ```

## Utilizzo

1. Esegui lo script principale per analizzare i dati:
    ```bash
    python src/main.py
    ```


