# 📧 Klasyfikator SPAM/HAM - Aplikacja Streamlit

Aplikacja webowa do klasyfikacji wiadomości tekstowych jako SPAM lub HAM (normalne wiadomości) z wykorzystaniem uczenia maszynowego.

## 📋 Wymagania

- Python 3.8+
- Biblioteki wymienione w `requirements.txt`

## 🚀 Instalacja i uruchomienie

### 1. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 2. Trenowanie modeli

Przed uruchomieniem aplikacji należy wytrenować modele:

```bash
python train_model.py
```

Ten skrypt:
- Utworzy katalog `models/`
- Wytrenuje 4 różne modele (Naive Bayes, Logistic Regression, SVM, Random Forest)
- Zapisze modele i vectorizer do plików `.joblib`
- Zapisze metryki do pliku JSON

### 3. Uruchomienie aplikacji

```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem: `http://localhost:8501`

## 📂 Struktura projektu

```
Webapp/
├── app.py              # Główna aplikacja Streamlit
├── train_model.py      # Skrypt do trenowania modeli
├── requirements.txt    # Zależności projektu
├── README.md           # Dokumentacja
└── models/             # Katalog z modelami (tworzony automatycznie)
    ├── vectorizer.joblib
    ├── naive_bayes.joblib
    ├── logistic_regression.joblib
    ├── svm.joblib
    ├── random_forest.joblib
    ├── metrics.json
    └── test_data.json
```

## 🎯 Funkcjonalności

### 1. Strona główna
- Opis projektu i informacje o zbiorze danych
- Wyjaśnienie użytych technik ML

### 2. Klasyfikator interaktywny
- Pole tekstowe do wprowadzania wiadomości
- Klasyfikacja w czasie rzeczywistym
- Wyświetlanie wyniku (SPAM/HAM)
- Wykres słupkowy prawdopodobieństw

### 3. Analiza modelu
- Metryki: Accuracy, Precision, Recall, F1-Score
- Macierz pomyłek (Confusion Matrix)
- Porównanie wszystkich modeli na wykresie

### 4. Przykłady demonstracyjne
- Gotowe przykłady wiadomości spam i ham
- Możliwość szybkiego testowania

## 🔧 Użyte technologie

- **Streamlit** - framework do tworzenia aplikacji webowych
- **scikit-learn** - modele ML i przetwarzanie tekstu
- **Plotly** - interaktywne wykresy
- **joblib** - serializacja modeli
- **pandas/numpy** - przetwarzanie danych

## 📊 Modele

| Model | Opis |
|-------|------|
| Naive Bayes | Probabilistyczny klasyfikator oparty na twierdzeniu Bayesa |
| Logistic Regression | Model liniowy z funkcją sigmoidalną |
| SVM | Maszyna wektorów nośnych z liniowym kernelem |
| Random Forest | Zespół drzew decyzyjnych |

## 📝 Autor

Projekt wykonany w ramach przedmiotu **Sztuczna Inteligencja**  
Semestr 7 | 2025/2026
