# Datathon: Datenbereinigung und Business Case Entwicklung
## Zalando E-Commerce Datensatz - Analyse und Handlungsempfehlungen

---

## TEIL 1: DATENBEREINIGUNG - STRATEGIE UND MASSNAHMEN

### 1.1 Identifizierte Datenqualitätsprobleme

#### Problem 1: Spaltenname-Fehler
- **Spalte**: `Shirtsyment`
- **Sollte sein**: `Payment` (Zahlungsmethode)
- **Maßnahme**: Spalte umbenennen

#### Problem 2: Inkonsistenzen in kategorischen Variablen

**Device-Variable:**
- `Phone` (1.231 Einträge) und `Mobile Phone` (2.765 Einträge) sind semantisch identisch
- `Computer` (1.634 Einträge) ist klar unterschieden
- **Maßnahme**: `Phone` in `Mobile Phone` zusammenführen

**Payment-Variable (Shirtsyment):**
- `COD` (365 Einträge) und `Cash on Delivery` (149 Einträge) sind identisch
- `CC` (273 Einträge) und `Credit Card` (1.501 Einträge) sind identisch
- `E wallet` (614 Einträge) und `UPI` (414 Einträge) sind separate Kategorien
- `Debit Card` (2.314 Einträge)
- **Maßnahme**: Standardisierung durchführen:
  - `COD` + `Cash on Delivery` → `Cash on Delivery`
  - `CC` + `Credit Card` → `Credit Card`

#### Problem 3: Fehlende Werte
| Variable | Fehlende Werte | Prozentsatz |
|----------|-----------------|------------|
| Visits | 264 | 4,69% |
| Distance | 251 | 4,46% |
| FaceTime | 255 | 4,53% |
| Coupons | 256 | 4,55% |
| NumOrders | 258 | 4,58% |
| LastOrder | 307 | 5,45% |

**Maßnahmen nach Variable:**
- **Visits, Coupons, NumOrders**: Imputation mit Median (da numerisch und nicht MCAR)
- **Distance**: Imputation mit Median pro Region (lokale Logistik)
- **FaceTime**: Imputation mit Median (ordinale Variable)
- **LastOrder**: Imputation mit Median (repräsentiert Inaktivität)

#### Problem 4: Duplikate
- **556 Duplikate** (außer CustID) identifiziert
- **Ursache**: Wahrscheinlich mehrere Transaktionen desselben Kunden
- **Maßnahme**: **NICHT entfernen** - diese sind legitime Mehrfach-Transaktionen
  - Stattdessen: Aggregation auf Kundenebene für bestimmte Analysen erwägen

#### Problem 5: Outlier
Signifikante Outlier in folgenden Variablen:
- **NoDevices**: 397 Outlier (7,05%)
- **Coupons**: 629 Outlier (11,17%)
- **NumOrders**: 703 Outlier (12,49%)
- **BasSize**: 948 Outlier (16,84%)
- **Returns**: 438 Outlier (7,78%)

**Maßnahme**: **NICHT entfernen** - diese sind legitime Geschäftsfälle (z.B. Vielkäufer)
- Stattdessen: Separate Analyse für Segmentierung durchführen

#### Problem 6: Konsistenzprüfung
- **Order vs BasSize**: Perfekte Konsistenz ✓
  - Order='Ja' → BasSize > 0 (100% Konsistenz)
  - Order='Nein' → BasSize = 0 (100% Konsistenz)
  - Korrelation: 0,927 (sehr stark)

---

### 1.2 Datenbereinigungsschritte (Reihenfolge)

#### Schritt 1: Spaltenumbenennung
```python
df.rename(columns={'Shirtsyment': 'Payment'}, inplace=True)
```

#### Schritt 2: Kategorische Standardisierung
```python
# Device standardisieren
df['Device'] = df['Device'].replace('Phone', 'Mobile Phone')

# Payment standardisieren
df['Payment'] = df['Payment'].replace({
    'Cash on Delivery': 'Cash on Delivery',  # Normalisierung
    'COD': 'Cash on Delivery',
    'Credit Card': 'Credit Card',
    'CC': 'Credit Card'
})
```

#### Schritt 3: Fehlende Werte behandeln
```python
# Numerische Variablen mit Median imputieren
numeric_missing = ['Visits', 'Coupons', 'NumOrders']
for col in numeric_missing:
    df[col].fillna(df[col].median(), inplace=True)

# Distance pro Region imputieren (lokale Logik)
for region in df['Region'].unique():
    region_median = df[df['Region'] == region]['Distance'].median()
    df.loc[(df['Region'] == region) & (df['Distance'].isna()), 'Distance'] = region_median

# FaceTime mit Median imputieren
df['FaceTime'].fillna(df['FaceTime'].median(), inplace=True)

# LastOrder mit Median imputieren
df['LastOrder'].fillna(df['LastOrder'].median(), inplace=True)
```

#### Schritt 4: Datentyp-Optimierung
```python
# Order in numerisch konvertieren (für Modellierung)
df['Order_Binary'] = (df['Order'] == 'Ja').astype(int)

# Complaints und Gender sind bereits binary
df['Complaints_Binary'] = df['Complaints'].astype(int)
df['Gender_Binary'] = (df['Gender'] == 'Male').astype(int)
```

#### Schritt 5: Qualitätssicherung
```python
# Nach Bereinigung prüfen
assert df.isnull().sum().sum() == 0, "Noch fehlende Werte vorhanden!"
assert df['Device'].nunique() == 2, "Device sollte 2 Kategorien haben"
assert df['Payment'].nunique() == 5, "Payment sollte 5 Kategorien haben"
```

---

## TEIL 2: BUSINESS CASE ENTWICKLUNG

### 2.1 Geschäftliche Ziele und Zielvariablen

Basierend auf der Aufgabenstellung des Head of eCommerce können folgende **Zielvariablen** identifiziert werden:

#### Zielmetrik 1: **Konversionsoptimierung** (PRIMARY)
- **Zielvariable**: `Order` (Binary: Ja/Nein)
- **Geschäftlicher Nutzen**: Direkte Umsatzsteigerung
- **Modelltyp**: Binary Classification
- **Aktuelle Konversionsrate**: 16,84% (948 von 5.630 Transaktionen)
- **Geschäftliche Fragen**:
  - Welche Kundenmerkmale führen zu höheren Konversionsraten?
  - Welche Geräte/Zahlungsmethoden sind konversionsfreundlich?
  - Wie wirken sich Visits und FaceTime auf Konversion aus?

**Geschäftlicher Impact**: 
- Jede 1% Steigerung der Konversionsrate = +56 zusätzliche Käufe pro 5.630 Transaktionen
- Bei durchschnittlichem Warenkorbwert (siehe unten) = signifikanter Umsatzzuwachs

---

#### Zielmetrik 2: **Kundenzufriedenheit** (SECONDARY)
- **Zielvariable**: `CSat` (Likert-Skala 1-5)
- **Geschäftlicher Nutzen**: Kundenbindung, Wiederholungskäufe, Reputation
- **Modelltyp**: Multi-Class Classification oder Regression
- **Aktuelle Verteilung**:
  - 1 (Sehr unzufrieden): 1.164 (20,7%)
  - 2 (Unzufrieden): 586 (10,4%)
  - 3 (Neutral): 1.698 (30,2%)
  - 4 (Zufrieden): 1.074 (19,1%)
  - 5 (Sehr zufrieden): 1.108 (19,7%)
- **Geschäftliche Fragen**:
  - Welche Faktoren beeinflussen die Kundenzufriedenheit?
  - Wie wirken sich Beschwerden und Rückgaben auf CSat aus?
  - Welche Kundengruppen sind am zufriedensten?

**Geschäftlicher Impact**: 
- Zufriedene Kunden (CSat 4-5) = 38,8% der Kundenbasis
- Unzufriedene Kunden (CSat 1-2) = 31,1% der Kundenbasis
- Verbesserung um 10% → +561 zufriedene Kunden

---

#### Zielmetrik 3: **Warenkorbwert-Optimierung** (SECONDARY)
- **Zielvariable**: `BasSize` (Kontinuierlich, EUR)
- **Geschäftlicher Nutzen**: Durchschnittlicher Bestellwert (AOV) erhöhen
- **Modelltyp**: Regression
- **Aktuelle Statistiken**:
  - Mittelwert: 63,47 EUR
  - Median: 0 EUR (viele Nicht-Käufer)
  - Max: 595 EUR
  - Nur bei Käufern (BasSize > 0): Durchschnitt ~380 EUR
- **Geschäftliche Fragen**:
  - Welche Kundenmerkmale führen zu höheren Warenkorbwerten?
  - Wie wirken sich Coupons auf BasSize aus?
  - Welche Produktkategorien haben höhere AOV?

**Geschäftlicher Impact**: 
- Durchschnittliche Steigerung um 10 EUR = +56.300 EUR Umsatz (bei 5.630 Transaktionen)

---

#### Zielmetrik 4: **Kundensegmentierung** (EXPLORATORY)
- **Zielvariable**: Cluster-Labels (selbst generiert)
- **Geschäftlicher Nutzen**: Personalisierte Marketing- und Produktstrategien
- **Modelltyp**: Clustering (K-Means, Hierarchical Clustering)
- **Segmentierungsdimensionen**:
  - Kaufverhalten (Visits, NumOrders, LastOrder)
  - Engagement (FaceTime, NoDevices)
  - Wirtschaftlicher Wert (BasSize, Returns)
  - Zufriedenheit (CSat, Complaints)

**Geschäftlicher Impact**: 
- Identifikation von High-Value-Kunden für Premium-Services
- Identifikation von At-Risk-Kunden für Retention-Kampagnen
- Personalisierte Empfehlungen pro Segment

---

#### Zielmetrik 5: **Rückgabequoten-Prognose** (EXPLORATORY)
- **Zielvariable**: `Returns` (Kontinuierlich, EUR) oder Returns_Binary (Hat Rückgaben ja/nein)
- **Geschäftlicher Nutzen**: Kostenreduktion, Logistik-Optimierung
- **Modelltyp**: Regression oder Binary Classification
- **Aktuelle Statistiken**:
  - Durchschnitt: 177,22 EUR
  - Median: 163,28 EUR
  - 438 Outlier (7,78%)
- **Geschäftliche Fragen**:
  - Welche Produktkategorien haben höhere Rückgabequoten?
  - Wie wirken sich Zahlungsmethoden auf Rückgaben aus?
  - Welche Kundengruppen sind Rückgabe-anfällig?

**Geschäftlicher Impact**: 
- Reduktion der Rückgabequote um 5% = Einsparung von ~44.400 EUR (bei 5.630 Transaktionen)

---

### 2.2 Empfohlene Modellierungsansätze

#### Modell 1: **Konversions-Prognose** (PRIMARY)
- **Problemtyp**: Binary Classification
- **Zielvariable**: `Order` (Ja/Nein)
- **Empfohlene Algorithmen**:
  - Logistic Regression (Baseline, interpretierbar)
  - Random Forest Classifier (hohe Genauigkeit)
  - Gradient Boosting (XGBoost/LightGBM für beste Performance)
- **Features**: Alle außer Order und BasSize
- **Evaluationsmetriken**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Business KPI**: Conversion Lift (wie viel % Steigerung durch Modell)

#### Modell 2: **Kundenzufriedenheits-Prognose** (SECONDARY)
- **Problemtyp**: Multi-Class Classification oder Regression
- **Zielvariable**: `CSat` (1-5)
- **Empfohlene Algorithmen**:
  - Ordinal Logistic Regression (respektiert Ordnung)
  - Random Forest Classifier (Multi-Class)
  - Gradient Boosting Classifier
- **Features**: Alle außer CSat und Order
- **Evaluationsmetriken**: Accuracy, Macro F1-Score, Confusion Matrix
- **Business KPI**: Anteil zufriedener Kunden (CSat >= 4)

#### Modell 3: **Kundensegmentierung** (EXPLORATORY)
- **Problemtyp**: Unsupervised Learning (Clustering)
- **Zielvariable**: Cluster-Labels (zu generieren)
- **Empfohlene Algorithmen**:
  - K-Means (einfach, interpretierbar)
  - Hierarchical Clustering (dendrogramm-basiert)
  - DBSCAN (dichte-basiert)
- **Features**: Skalierte numerische Features (Visits, FaceTime, BasSize, CSat, NumOrders, Returns)
- **Evaluationsmetriken**: Silhouette Score, Davies-Bouldin Index, Elbow Method
- **Business KPI**: Segment-Profile und Actionability

---

### 2.3 Feature-Engineering-Möglichkeiten

#### Neue Features aus bestehenden Variablen:

1. **Engagement-Index**
   ```
   Engagement = (Visits + FaceTime + NoDevices) / 3
   ```
   - Misst Kundenaktivität auf der Plattform

2. **Loyalitäts-Score**
   ```
   Loyalty = (NumOrders * 10 + (1 / (LastOrder + 1))) / 2
   ```
   - Höher = häufiger Käufer, kürzliche Aktivität

3. **Rückgabe-Quote**
   ```
   Return_Rate = Returns / (BasSize + 1)  # +1 zur Vermeidung von Division durch 0
   ```
   - Misst Rückgabeverhalten relativ zu Kaufwert

4. **Coupon-Sensitivität**
   ```
   Coupon_Sensitivity = Coupons / (NumOrders + 1)
   ```
   - Wie stark nutzt der Kunde Rabatte?

5. **Beschwerde-Flag**
   ```
   Has_Complaints = Complaints (bereits vorhanden, aber als Feature nutzen)
   ```
   - Binärer Indikator für Kundenprobleme

6. **Device-Konsistenz**
   ```
   Device_Consistency = 1 if Device == 'Mobile Phone' else 0
   ```
   - Mobile-First Trend in E-Commerce

7. **Zahlungsmethoden-Kategorie**
   ```
   Payment_Category:
   - Digital (E-wallet, UPI)
   - Card (Credit Card, Debit Card)
   - COD (Cash on Delivery)
   ```
   - Gruppierung nach Zahlungstyp

---

### 2.4 Geschäftliche Erfolgskriterien

| Modell | Zielmetrik | Erfolgskriterium | Business Impact |
|--------|-----------|-----------------|-----------------|
| **Konversion** | F1-Score | > 0,70 | Mindestens 70% der Käufer korrekt identifizieren |
| **Konversion** | Precision | > 0,60 | Mindestens 60% der Vorhersagen sind korrekt |
| **Zufriedenheit** | Accuracy | > 0,50 | Besser als Zufallsvorhersage (20% Baseline) |
| **Zufriedenheit** | Macro F1 | > 0,40 | Balancierte Performance über alle Klassen |
| **Segmentierung** | Silhouette | > 0,40 | Gute Cluster-Separation |
| **Segmentierung** | Interpretierbarkeit | Actionable Profiles | Klare Geschäftsimplikationen pro Segment |

---

### 2.5 Implementierungs- und Integrationsstrategie

#### Deployment-Szenarien:

1. **Echtzeit-Konversions-Prognose**
   - Integration in Checkout-Prozess
   - Automatische Angebote für At-Risk-Kunden
   - A/B-Testing von Interventionen

2. **Kundenzufriedenheits-Monitoring**
   - Automatische Alerts bei CSat-Risiko
   - Proaktive Kundenservice-Interventionen
   - Feedback-Loop für Modell-Verbesserung

3. **Personalisierte Empfehlungen**
   - Segment-basierte Produktempfehlungen
   - Dynamische Pricing pro Segment
   - Personalisierte Marketing-Kampagnen

4. **Operative Optimierungen**
   - Logistik-Routing basierend auf Distance-Prognosen
   - Zahlungsmethoden-Optimierung pro Region
   - Inventory-Management basierend auf Kategorie-Prognosen

---

## ZUSAMMENFASSUNG: DATENBEREINIGUNG UND BUSINESS CASE

### Datenbereinigungszusammenfassung:
✓ **6 Datenqualitätsprobleme** identifiziert und Lösungen definiert
✓ **Fehlende Werte** (4-5%) mit Imputation behandelt
✓ **Kategorische Standardisierung** durchgeführt
✓ **Duplikate** als legitim erkannt und beibehalten
✓ **Konsistenz** zwischen Order und BasSize verifiziert

### Business Case Zusammenfassung:
✓ **5 Zielvariablen** identifiziert (1 primär, 4 sekundär/explorativ)
✓ **3 Modellierungsansätze** definiert (Classification, Regression, Clustering)
✓ **7 Feature-Engineering-Optionen** vorgeschlagen
✓ **Erfolgskriterien** für jedes Modell festgelegt
✓ **Implementierungsszenarien** skizziert

### Geschäftlicher Mehrwert:
- **Konversionsoptimierung**: +1-5% Steigerung = +56-280 zusätzliche Käufe
- **Kundenzufriedenheit**: +10% Verbesserung = +561 zufriedene Kunden
- **AOV-Steigerung**: +10 EUR = +56.300 EUR Umsatz
- **Rückgabe-Reduktion**: -5% = -44.400 EUR Einsparung
- **Segmentierung**: Personalisierung für höhere Lifetime Value

