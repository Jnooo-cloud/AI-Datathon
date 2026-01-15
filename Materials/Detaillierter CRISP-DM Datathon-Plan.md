# Detaillierter CRISP-DM Datathon-Plan

Dieser Plan bietet eine strukturierte Vorgehensweise für den Datathon, basierend auf dem CRISP-DM-Modell und den in deinem Repository bereitgestellten Materialien.

## Phase 1: Business Understanding (ca. 1-2 Stunden)

**Ziel:** Das Geschäftsproblem und die Erfolgskriterien vollständig verstehen.

| Schritt | Aktion | Tools & Konzepte |
|---|---|---|
| 1 | **Business Case analysieren** | `Process checklist` zur Identifizierung des Problemtyps (Regression, Klassifikation, Clustering). |
| 2 | **Erfolgsmetriken definieren** | Je nach Problemtyp: R-Squared, F1-Score, Silhouette-Score. Berücksichtigung des Business-Impacts. |
| 3 | **Business-Fragen formulieren** | Übersetzung des Problems in spezifische, beantwortbare Fragen. |
| 4 | **Dokumentation** | Erstellung eines "Business Understanding"-Abschnitts im Jupyter Notebook zur Dokumentation der Ergebnisse. |

## Phase 2: Data Understanding (ca. 2-3 Stunden)

**Ziel:** Vertrautheit mit den Daten erlangen, Datenqualitätsprobleme identifizieren und erste Einblicke gewinnen.

| Schritt | Aktion | Tools & Konzepte |
|---|---|---|
| 1 | **Daten laden und inspizieren** | `pandas`: `df.info()`, `df.head()`, `df.describe()` |
| 2 | **Explorative Datenanalyse (EDA)** | `seaborn` & `matplotlib`: `sns.histplot`, `sns.boxplot`, `sns.scatterplot`, `sns.countplot`, `sns.heatmap` |
| 3 | **Datenqualitätsprüfung** | `pandas`: `df.isnull().sum()` |
| 4 | **Dokumentation** | Erstellung eines "Data Understanding"-Abschnitts mit allen Erkenntnissen und Visualisierungen. |

## Phase 3: Data Preparation (ca. 3-4 Stunden)

**Ziel:** Die Daten für die Modellierung bereinigen, transformieren und vorbereiten.

| Schritt | Aktion | Tools & Konzepte |
|---|---|---|
| 1 | **Datenbereinigung** | `pandas`: `fillna()`, `dropna()`. Fortgeschritten: `KNNImputer` |
| 2 | **Feature Engineering** | `pandas`: `get_dummies` für One-Hot-Encoding |
| 3 | **Datentransformation** | `sklearn.preprocessing`: `StandardScaler`, `numpy`: `log()` |
| 4 | **Train-Test-Split** | `sklearn.model_selection`: `train_test_split` |
| 5 | **Dokumentation** | Erstellung eines "Data Preparation"-Abschnitts, der jeden Schritt und seine Begründung dokumentiert. |

## Phase 4: Modeling (ca. 2-3 Stunden)

**Ziel:** Modelle trainieren, die das Geschäftsproblem lösen.

| Schritt | Aktion | Tools & Konzepte |
|---|---|---|
| 1 | **Modell(e) auswählen** | Basierend auf dem Problemtyp: `LinearRegression`, `LogisticRegression`, `RandomForestClassifier`, `KMeans` etc. |
| 2 | **Modell(e) trainieren** | `sklearn`: `.fit()`-Methode auf den Trainingsdaten |
| 3 | **Hyperparameter-Tuning (optional)** | `GridSearchCV` oder `RandomizedSearchCV` zur Optimierung der Modellparameter. |
| 4 | **Dokumentation** | Erstellung eines "Modeling"-Abschnitts, der die Modellauswahl und den Trainingsprozess beschreibt. |

## Phase 5: Evaluation (ca. 1-2 Stunden)

**Ziel:** Die Leistung der Modelle bewerten und das beste Modell auswählen.

| Schritt | Aktion | Tools & Konzepte |
|---|---|---|
| 1 | **Modellleistung bewerten** | `sklearn.metrics`: `confusion_matrix`, `classification_report`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score` |
| 2 | **Modelle vergleichen** | Vergleich der Leistung verschiedener Modelle anhand der definierten Metriken. |
| 3 | **Ergebnisse interpretieren** | `SHAP` zur Erklärung der Modellvorhersagen und zur Identifizierung der wichtigsten Features. |
| 4 | **Dokumentation** | Erstellung eines "Evaluation"-Abschnitts mit den Ergebnissen der Modellbewertung. |

## Phase 6: Deployment & Ergebnispräsentation (ca. 1-2 Stunden)

**Ziel:** Die Ergebnisse verständlich aufbereiten und präsentieren.

| Schritt | Aktion | Tools & Konzepte |
|---|---|---|
| 1 | **Ergebnisse zusammenfassen** | Klare und prägnante Zusammenfassung der wichtigsten Erkenntnisse und Handlungsempfehlungen. |
| 2 | **Visualisierungen erstellen** | Erstellung von aussagekräftigen Diagrammen und Grafiken zur Unterstützung der Präsentation. |
| 3 | **Präsentation vorbereiten** | Erstellung einer Präsentation, die die Vorgehensweise, die Ergebnisse und den Business Value darstellt. |
| 4 | **(Optional) Interaktive Demo** | `Gradio` zur Erstellung einer einfachen Benutzeroberfläche für das Modell. |
