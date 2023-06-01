#en utilisant l'algorithme de forêt aléatoire de Scikit-Learn :

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Lecture des données
data = pd.read_csv('credit_card_data.csv')

# Exploration des données
print(data.head())
print(data.describe())

# Visualisation de la distribution de classes
sns.countplot(x='Class', data=data)
plt.title('Distribution de classes')
plt.show()

# Division des données en ensembles de formation et de test
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Formation du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Matrice de confusion:\n', conf_mat)
print('Précision:', precision)
print('Rappel:', recall)
print('Score F1:', f1)
print('Exactitude:', accuracy)

