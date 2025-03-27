import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score

df = pd.read_csv("matches.csv")

X = df.drop(['Jogador', 'Jogo', 'Data', 'Gol'], axis=1)
y = df['Gol']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier()
svm = SVC()
logreg = LogisticRegression()
dt = DecisionTreeClassifier(max_depth=3)
iso_forest = IsolationForest(contamination=0.1)
one_class_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
kmeans = KMeans(n_clusters=3, random_state=42)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
logreg.fit(X_train, y_train)
dt.fit(X_train, y_train)
iso_forest.fit(X_train)
one_class_svm.fit(X_train)
kmeans.fit(X_train)

rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
logreg_pred = logreg.predict(X_test)
dt_pred = dt.predict(X_test)

iso_forest_pred = iso_forest.predict(X_test)
iso_forest_pred = [1 if x == -1 else 0 for x in iso_forest_pred]
one_class_svm_pred = one_class_svm.predict(X_test)
one_class_svm_pred = [1 if x == -1 else 0 for x in one_class_svm_pred]

kmeans_pred = kmeans.predict(X_test)

classification_scores = {
    'Random Forest Accuracy': accuracy_score(y_test, rf_pred),
    'SVM Accuracy': accuracy_score(y_test, svm_pred),
    'Logistic Regression Accuracy': accuracy_score(y_test, logreg_pred),
    'Decision Tree Accuracy': accuracy_score(y_test, dt_pred)
}

tree_rules = export_text(dt, feature_names=list(X.columns))

sil_score = silhouette_score(X_test, kmeans_pred)

print("Classification Scores:")
for model, score in classification_scores.items():
    print(f"{model}: {score}")

print("\nDecision Tree Rules:")
print(tree_rules)

print("\nAnomaly Detection Results:")
print(f"Isolation Forest Anomalies: {sum(iso_forest_pred)}")
print(f"One-Class SVM Anomalies: {sum(one_class_svm_pred)}")

print("\nClustering Results:")
print(f"Silhouette Score for KMeans: {sil_score}")
