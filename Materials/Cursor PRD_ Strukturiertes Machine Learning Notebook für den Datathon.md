# Cursor PRD: Strukturiertes Machine Learning Notebook für den Datathon

## Executive Summary

Dieses PRD definiert die Anforderungen an die KI-gestützte Code-Generierung für ein Jupyter Notebook im Datathon-Kontext. Das Ziel ist die Erstellung eines **lean, dokumentierten und vollständig nachvollziehbaren** Notebooks, das dem CRISP-DM-Prozess folgt und ausschließlich die in den Vorlesungsmaterialien behandelten ML-Konzepte verwendet.

---

## 1. Allgemeine Prinzipien

### 1.1 Code-Philosophie

Das Notebook soll nach folgenden Prinzipien entwickelt werden:

- **Lean Code:** Prägnant, effizient und ohne unnötige Komplexität. Jede Zeile Code muss einen klaren Zweck erfüllen.
- **Explizit vor Implizit:** Jeder Schritt muss explizit implementiert und begründet werden. Automatisierte Refactoring-Tools oder "Lazy Predict"-Bibliotheken sind nicht erlaubt.
- **Dokumentation First:** Jeder Code-Block wird durch Markdown-Zellen und klare Kommentare dokumentiert.
- **Nachvollziehbarkeit:** Der Code muss so geschrieben sein, dass ein Leser die Logik und die Entscheidungen verstehen kann, ohne externe Ressourcen zu konsultieren.

### 1.2 Formatierungsrichtlinien

| Richtlinie | Anforderung |
|---|---|
| **Kommentare** | Klare, prägnante Kommentare in Deutsch. Keine Emojis. |
| **Variablennamen** | Aussagekräftig und in Englisch (z.B. `df_clean`, `X_train`, `y_test`). |
| **Zellstruktur** | Jeder logische Schritt in einer separaten Zelle oder einer logischen Gruppe. |
| **Markdown-Dokumentation** | Vor jedem Abschnitt eine Markdown-Zelle mit Überschrift und Erläuterung. |

---

## 2. Notebook-Struktur nach CRISP-DM

Das Notebook muss folgende Struktur aufweisen:

### 2.1 Phase 1: Business Understanding

**Markdown-Zelle:**
- Titel: "Business Understanding"
- Inhalt:
  - Beschreibung des Geschäftsproblems
  - Identifizierung der Zielvariable
  - Bestimmung des Problemtyps (Regression, Klassifikation, Clustering)
  - Definition der Erfolgsmetriken
  - Formulierung der Business-Fragen

**Code-Zelle (optional):**
- Kommentar: "Initialisierung und Konfiguration"
- Imports: Alle notwendigen Bibliotheken
- Konfiguration: z.B. `pd.set_option('display.max_columns', None)`

### 2.2 Phase 2: Data Understanding

**Markdown-Zelle:**
- Titel: "Data Understanding"
- Inhalt: Erläuterung der EDA-Schritte

**Code-Zelle 1: Datensatz laden**
```python
# Datensatz laden
df = pd.read_csv('path/to/dataset.csv')

# Überblick über den Datensatz
print(f"Datensatz-Form: {df.shape}")
print(f"Anzahl der Zeilen: {df.shape[0]}")
print(f"Anzahl der Spalten: {df.shape[1]}")
```

**Code-Zelle 2: Datentypen und fehlende Werte**
```python
# Datentypen und fehlende Werte
print("Datentypen und fehlende Werte:")
print(df.info())

# Zusammenfassung der fehlenden Werte
print("\nFehlende Werte pro Spalte:")
print(df.isnull().sum())
```

**Code-Zelle 3: Grundlegende Statistiken**
```python
# Grundlegende statistische Zusammenfassung
print("Statistische Zusammenfassung:")
print(df.describe())
```

**Code-Zelle 4: Explorative Visualisierungen**
```python
# Visualisierung der Verteilungen (numerische Variablen)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
fig.suptitle('Verteilungen numerischer Variablen', fontsize=14)

# Beispiel: Histogramme für die ersten 4 numerischen Spalten
numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
for idx, col in enumerate(numeric_cols):
    ax = axes[idx // 2, idx % 2]
    df[col].hist(bins=30, ax=ax, edgecolor='black')
    ax.set_title(f'Verteilung von {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Häufigkeit')

plt.tight_layout()
plt.show()
```

**Code-Zelle 5: Korrelationsanalyse**
```python
# Korrelationsmatrix für numerische Variablen
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Visualisierung der Korrelationsmatrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Korrelationsmatrix')
plt.show()
```

**Markdown-Zelle:**
- Zusammenfassung der EDA-Erkenntnisse
- Identifizierte Probleme (fehlende Werte, Ausreißer, etc.)

### 2.3 Phase 3: Data Preparation

**Markdown-Zelle:**
- Titel: "Data Preparation"
- Inhalt: Übersicht über die Datenbereinigungsschritte

**Code-Zelle 1: Umgang mit fehlenden Werten**
```python
# Überprüfung fehlender Werte
print("Fehlende Werte vor der Behandlung:")
print(df.isnull().sum())

# Strategie zur Behandlung fehlender Werte
# Option 1: Zeilen mit fehlenden Werten löschen (wenn < 5%)
# df = df.dropna()

# Option 2: Fehlende Werte mit Mittelwert/Median füllen
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print("\nFehlende Werte nach der Behandlung:")
print(df.isnull().sum())
```

**Code-Zelle 2: Ausreißer identifizieren und behandeln**
```python
# Ausreißer identifizieren (IQR-Methode)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Definition der Grenzen für Ausreißer
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Anzahl der Ausreißer pro Spalte
print("Anzahl der Ausreißer pro Spalte:")
for col in df.select_dtypes(include=[np.number]).columns:
    outliers = ((df[col] < lower_bound[col]) | (df[col] > upper_bound[col])).sum()
    print(f"{col}: {outliers}")

# Ausreißer entfernen (optional)
# df = df[(df >= lower_bound) & (df <= upper_bound)].dropna()
```

**Code-Zelle 3: Kategorische Variablen kodieren**
```python
# Identifizierung kategorischer Variablen
categorical_cols = df.select_dtypes(include=['object']).columns

print(f"Kategorische Variablen: {list(categorical_cols)}")

# One-Hot-Encoding für kategorische Variablen
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"\nForm nach Encoding: {df_encoded.shape}")
print(f"Neue Spalten: {list(df_encoded.columns)}")
```

**Code-Zelle 4: Feature Scaling**
```python
# Standardisierung der numerischen Features
from sklearn.preprocessing import StandardScaler

# Auswahl der numerischen Spalten (ohne Zielvariabl)
numeric_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
# Entfernen der Zielvariable, falls vorhanden
if 'target' in numeric_features:
    numeric_features.remove('target')

# Standardisierung
scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

print("Standardisierung abgeschlossen")
print(f"Mittelwert nach Skalierung: {df_scaled[numeric_features].mean().mean():.6f}")
print(f"Standardabweichung nach Skalierung: {df_scaled[numeric_features].std().mean():.6f}")
```

**Code-Zelle 5: Train-Test-Split**
```python
# Aufteilung in Features und Zielvariabl
X = df_scaled.drop('target', axis=1)  # Ersetze 'target' mit der echten Zielvariable
y = df_scaled['target']

# Train-Test-Split (80-20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Größe des Trainingssatzes: {X_train.shape}")
print(f"Größe des Testsatzes: {X_test.shape}")
```

### 2.4 Phase 4: Modeling

**Markdown-Zelle:**
- Titel: "Modeling"
- Inhalt: Begründung für die Modellauswahl

**Code-Zelle 1: Modelltraining (Beispiel: Logistische Regression)**
```python
# Modelltraining: Logistische Regression
from sklearn.linear_model import LogisticRegression

# Modell initialisieren
model = LogisticRegression(random_state=42, max_iter=1000)

# Modell trainieren
model.fit(X_train, y_train)

print("Modell erfolgreich trainiert")
print(f"Modellparameter: {model.get_params()}")
```

**Code-Zelle 2: Vorhersagen**
```python
# Vorhersagen auf Trainings- und Testdaten
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Vorhersagen abgeschlossen")
```

### 2.5 Phase 5: Evaluation

**Markdown-Zelle:**
- Titel: "Evaluation"
- Inhalt: Erläuterung der Evaluierungsmetriken

**Code-Zelle 1: Klassifikationsmetriken**
```python
# Evaluierung des Modells
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Accuracy und F1-Score
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
```

**Code-Zelle 2: Visualisierung der Confusion Matrix**
```python
# Visualisierung der Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Klasse 0', 'Klasse 1'], 
            yticklabels=['Klasse 0', 'Klasse 1'])
plt.title('Confusion Matrix')
plt.ylabel('Tatsächliche Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.show()
```

### 2.6 Phase 6: Ergebnispräsentation

**Markdown-Zelle:**
- Titel: "Zusammenfassung und Handlungsempfehlungen"
- Inhalt:
  - Zusammenfassung der Ergebnisse
  - Wichtigste Erkenntnisse
  - Handlungsempfehlungen für das Business

---

## 3. Detaillierte Anforderungen an die Code-Generierung

### 3.1 Imports und Bibliotheken

Die folgenden Bibliotheken sollen verwendet werden (basierend auf den Vorlesungsmaterialien):

```python
# Data Handling
import pandas as pd
import numpy as np
import os
import glob

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Modeling
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

# Statistics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### 3.2 Fehlerbehandlung und Validierung

- **Datenvalidierung:** Der Code muss überprüfen, ob der Datensatz korrekt geladen wurde und die erwartete Struktur hat.
- **Fehlerbehandlung:** Potenzielle Fehler (z.B. fehlende Spalten, Typ-Fehler) sollten abgefangen und aussagekräftige Fehlermeldungen ausgegeben werden.

### 3.3 Kommentierung und Dokumentation

Jeder Code-Block muss folgende Struktur haben:

```python
# Kurze Beschreibung des Schritts
# Erläuterung der Logik, falls nicht offensichtlich

code_here = "implementation"

# Ausgabe oder Überprüfung des Ergebnisses
print(f"Ergebnis: {result}")
```

### 3.4 Visualisierungen

- **Konsistente Formatierung:** Alle Visualisierungen sollten ein einheitliches Aussehen haben.
- **Aussagekräftige Labels:** Alle Achsen und Legenden sollten klar beschriftet sein.
- **Größe und Auflösung:** Figuren sollten eine angemessene Größe haben (z.B. `figsize=(12, 8)`).

---

## 4. Spezifische Anforderungen für verschiedene Problemtypen

### 4.1 Regression

- **Modelle:** `LinearRegression`, `Ridge`, `Lasso`
- **Evaluierungsmetriken:** `R²`, `RMSE`, `MAE`
- **Visualisierungen:** Scatter-Plots (Actual vs. Predicted), Residual-Plots

### 4.2 Klassifikation

- **Modelle:** `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `KNN`
- **Evaluierungsmetriken:** `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`
- **Visualisierungen:** Confusion Matrix, Classification Report, ROC-Kurve (optional)

### 4.3 Clustering

- **Modelle:** `KMeans`
- **Evaluierungsmetriken:** `Silhouette Score`, `Inertia` (Elbow Method)
- **Visualisierungen:** Elbow-Kurve, Cluster-Visualisierungen

---

## 5. Best Practices und Anti-Patterns

### 5.1 Best Practices

| Best Practice | Beschreibung |
|---|---|
| **Reproduzierbarkeit** | Verwendung von `random_state=42` für alle stochastischen Operationen. |
| **Datenintegrität** | Keine Modifizierung des ursprünglichen DataFrames; stattdessen Kopien erstellen. |
| **Explizite Schritte** | Jeder Schritt wird explizit implementiert, nicht durch Automatisierung versteckt. |
| **Validierung** | Überprüfung der Daten nach jedem Transformationsschritt. |

### 5.2 Anti-Patterns (zu vermeiden)

| Anti-Pattern | Grund zur Vermeidung |
|---|---|
| **Lazy Predict** | Versteckt die Modellauswahl; nicht nachvollziehbar. |
| **Automatisiertes Refactoring** | Kann zu unerwartetem Verhalten führen. |
| **Fehlende Dokumentation** | Code wird unverständlich. |
| **Hardcodierte Werte** | Reduziert die Flexibilität und Wartbarkeit. |
| **Keine Validierung** | Kann zu fehlerhaften Ergebnissen führen. |

---

## 6. Checkliste für die Notebook-Erstellung

- [ ] Business Understanding: Problem und Metriken klar definiert
- [ ] Data Understanding: EDA durchgeführt und dokumentiert
- [ ] Data Preparation: Alle Bereinigungsschritte explizit implementiert
- [ ] Modeling: Modellauswahl begründet und trainiert
- [ ] Evaluation: Metriken berechnet und visualisiert
- [ ] Ergebnispräsentation: Zusammenfassung und Handlungsempfehlungen
- [ ] Code-Qualität: Lean, dokumentiert, nachvollziehbar
- [ ] Reproduzierbarkeit: `random_state` gesetzt, Daten validiert
- [ ] Keine Emojis: Im Code und in Kommentaren
- [ ] Markdown-Dokumentation: Vor jedem Abschnitt vorhanden

---

## 7. Beispiel: Vollständige Zellsequenz

### Markdown-Zelle
```markdown
## 2.3 Datenbereinigung

In diesem Abschnitt werden wir die Daten bereinigen, indem wir:
1. Fehlende Werte behandeln
2. Ausreißer identifizieren und entfernen
3. Kategorische Variablen kodieren
```

### Code-Zelle
```python
# Schritt 1: Fehlende Werte behandeln
print("Fehlende Werte vor der Behandlung:")
print(df.isnull().sum())

# Für numerische Spalten: Mittelwert verwenden
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\nFehlende Werte nach der Behandlung:")
print(df.isnull().sum())
```

---

## 8. Kommunikation mit Cursor

Wenn du Cursor verwendest, um Code zu generieren, verwende folgende Prompts:

**Beispiel-Prompt:**
```
Generiere eine Python-Zelle für die Explorative Datenanalyse (EDA) eines Datensatzes. 
Die Zelle soll:
1. Die Verteilung numerischer Variablen mit Histogrammen visualisieren
2. Eine Korrelationsmatrix berechnen und als Heatmap darstellen
3. Kategorische Variablen mit Countplots visualisieren

Anforderungen:
- Lean Code ohne unnötige Komplexität
- Klare Kommentare in Deutsch
- Keine Emojis
- Verwendung von matplotlib und seaborn
```

---

## 9. Fazit

Dieses PRD definiert einen strukturierten Ansatz zur Erstellung eines Machine-Learning-Notebooks für den Datathon. Der Fokus liegt auf **Klarheit, Nachvollziehbarkeit und Explizitheit**. Durch die Einhaltung dieser Anforderungen wird ein hochwertiges, wartbares und verständliches Notebook entstehen, das die Anforderungen des Datathons erfüllt.
