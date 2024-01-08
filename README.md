# Progetto NLP Code Generator -

Questo progetto utilizza il dataset Django per addestrare un modello di linguaggio naturale attraverso la libreria Transformers. Il modello apprende a generare codice Python in risposta a input di linguaggio naturale.

## Requisiti

Prima di iniziare, assicurati di avere installati i seguenti requisiti:

Python 3.10 o successivi
```bash
pip install -r requirements.txt
```

## Avvio dell'addestramento, testing o valutazione
```bash
python main.py
```
Nel file main.py, viene richiesto di inserire un numero corrispondente all'azione desiderata:
- 0 per l'addestramento
- 1 per il testing
- 2 per la valutazione del modello
- 3 per avere una demo delle risposte del modello

## Avvio dell'addestramento con GPU
Per avviare l'addestramento con accelerazione sulla GPU bisogna eseguire la sequente modifica:
```bash
fp16 = True
```
## Contributi
Siamo aperti a contributi! Sentiti libero di aprire issue o pull request per migliorare questo progetto.

