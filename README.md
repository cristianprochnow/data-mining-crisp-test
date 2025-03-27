# Data Mining CRISP-DM Method

Code for Data Mining class with CRISP method for N1 grade.

# Crew

* Cristian Prochnow
* Gustavo Henrique Dias

# Run

```bash
$ pip install -r requirements.txt
$ python3 script.py
```

# About

We use CSV data from `matches.csv` to extract some data from matches stats from it. All this data were randomly generated for test purposes only.

When running script, process will output stats about accuracy from data within CSV, that will be something like below.

```
Classification Scores:
Random Forest Accuracy: 0.5714285714285714
SVM Accuracy: 0.5714285714285714
Logistic Regression Accuracy: 0.5714285714285714
Decision Tree Accuracy: 0.5

Decision Tree Rules:
|--- Minutos Jogados <= 0.32
|   |--- Passes Certos (%) <= 0.67
|   |   |--- Minutos Jogados <= -0.40
|   |   |   |--- class: 0
|   |   |--- Minutos Jogados >  -0.40
|   |   |   |--- class: 0
|   |--- Passes Certos (%) >  0.67
|   |   |--- class: 1
|--- Minutos Jogados >  0.32
|   |--- Desarmes <= 1.11
|   |   |--- class: 0
|   |--- Desarmes >  1.11
|   |   |--- Distância Percorrida (km) <= 0.08
|   |   |   |--- class: 0
|   |   |--- Distância Percorrida (km) >  0.08
|   |   |   |--- class: 1


Anomaly Detection Results:
Isolation Forest Anomalies: 1
One-Class SVM Anomalies: 6

Clustering Results:
Silhouette Score for KMeans: 0.1306329110140733
```