"""
Aplikacja Streamlit do klasyfikacji wiadomości SPAM/HAM
Autor: Projekt AI - Semestr 7
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import pandas as pd, seaborn as sns, re, spacy
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
import scipy.sparse
import joblib
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
import warnings

# Ścieżka bazowa aplikacji
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Konfiguracja strony
st.set_page_config(
    page_title="Klasyfikator SPAM/HAM",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-spam {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .result-ham {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Funkcje pomocnicze
@st.cache_resource
def load_model():
    """Ładuje model LightGBM i vectorizer."""
    model = None
    vectorizer = None
    metrics = {}
    
    try:
        # Ładowanie vectorizera
        vectorizer_path = os.path.join(APP_DIR, 'tfidf_vectorizer.pkl')
        vectorizer = joblib.load(vectorizer_path)
        
        # Sprawdzenie czy vectorizer jest fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            st.error("Vectorizer nie jest fitted! Upewnij się, że został zapisany po wykonaniu fit_transform.")
            return None, None, {}
        
        # Ładowanie modelu LightGBM
        model_path = os.path.join(APP_DIR, 'tuned_LightGBM_model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        
        # Metryki - można dodać ręcznie lub załadować z pliku jeśli istnieje
        metrics_path = os.path.join(APP_DIR, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu: {e}")
        
    return model, vectorizer, metrics


def predict_message(message, model, vectorizer):
    """Klasyfikuje pojedynczą wiadomość."""
    # Przetwarzanie tekstu
    processed_message = preprocessing(message)
    
    # Wektoryzacja
    message_tfidf = vectorizer.transform([processed_message])
    
    # Predykcja
    prediction = model.predict(message_tfidf)[0]
    
    # Prawdopodobieństwa (jeśli model je wspiera)
    try:
        probabilities = model.predict_proba(message_tfidf)[0]
        prob_ham = probabilities[0]
        prob_spam = probabilities[1]
    except AttributeError:
        decision = model.decision_function(message_tfidf)[0]
        prob_spam = 1 / (1 + np.exp(-decision))  # Sigmoid
        prob_ham = 1 - prob_spam
    
    return prediction, prob_ham, prob_spam


def create_probability_chart(prob_ham, prob_spam):
    """Tworzy wykres słupkowy prawdopodobieństw."""
    fig = go.Figure(data=[
        go.Bar(
            x=['HAM (Normalna)', 'SPAM'],
            y=[prob_ham * 100, prob_spam * 100],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f'{prob_ham*100:.1f}%', f'{prob_spam*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prawdopodobieństwo klasyfikacji',
        xaxis_title='Klasa',
        yaxis_title='Prawdopodobieństwo (%)',
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )
    
    return fig


def create_confusion_matrix_plot(cm, title):
    """Tworzy wizualizację macierzy pomyłek."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['HAM (Pred)', 'SPAM (Pred)'],
        y=['HAM (True)', 'SPAM (True)'],
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Przewidywana klasa',
        yaxis_title='Rzeczywista klasa',
        height=400
    )
    
    return fig


def create_metrics_comparison_chart(metrics):
    """Tworzy wykres porównawczy metryk dla różnych modeli."""
    models_names = list(metrics.keys())
    
    accuracy = [metrics[m]['accuracy'] * 100 for m in models_names]
    precision = [metrics[m]['precision'] * 100 for m in models_names]
    recall = [metrics[m]['recall'] * 100 for m in models_names]
    f1 = [metrics[m]['f1_score'] * 100 for m in models_names]
    
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models_names, y=accuracy, marker_color='#3498db'),
        go.Bar(name='Precision', x=models_names, y=precision, marker_color='#2ecc71'),
        go.Bar(name='Recall', x=models_names, y=recall, marker_color='#e74c3c'),
        go.Bar(name='F1-Score', x=models_names, y=f1, marker_color='#9b59b6')
    ])
    
    fig.update_layout(
        title='Porównanie metryk dla różnych modeli',
        xaxis_title='Model',
        yaxis_title='Wartość (%)',
        barmode='group',
        yaxis_range=[0, 100],
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


# Przykładowe wiadomości
EXAMPLE_SPAM = [
    "<html><body><font face='Arial'>FREE MONEY! Click here NOW to claim your $1000 cash reward! Limited time offer! Visit www.freecash.com</font></body></html>",
    "<table><tr><td><font color='red'>CONGRATULATIONS!</font></td></tr></table> You have WON a FREE iPhone! Click this link immediately to claim your prize before it expires!",
    "<br><br><font face='Helvetica'>Dear winner, you have been selected to receive a special offer! Get 50% OFF on all products! Click here: http://specialoffer.com</font><br>",
    "!!!URGENT!!! Your account will be SUSPENDED unless you verify your information immediately. Click here now to avoid losing access: http://verify-account.net",
    "<html><head></head><body><table><tr><td><font size='5' color='blue'>AMAZING DEAL!</font></td></tr></table>Make $5000 per week from home! No experience needed! Email us at: easy-money@rich.com</body></html>"
]

EXAMPLE_HAM = [
    "Hi John, thanks for sending the report. I reviewed the quarterly numbers and they look good. Can we schedule a meeting next week to discuss the budget?",
    "The Linux kernel update has been released. You can download the RPM package from the mailing list. Please test it in your development environment before deployment.",
    "Meeting reminder: Team standup at 10am tomorrow in conference room B. Please bring your status updates and any blockers you're facing.",
    "I've pushed the code changes to the repository. The pull request includes bug fixes and new unit tests. Could you review it when you have time?",
    "Just wanted to follow up on our discussion from yesterday. I think we should proceed with option A for the database migration. Let me know your thoughts."
]


# Główna aplikacja
def main():
    # Sidebar - nawigacja
    st.sidebar.title("Nawigacja")
    page = st.sidebar.radio(
        "Wybierz stronę:",
        ["Strona główna", "Klasyfikator", "Analiza modelu", "Przykłady"]
    )
    
    # Ładowanie modelu
    model, vectorizer, metrics = load_model()
    
    if model is None or vectorizer is None:
        st.error("Model nie został załadowany. Wytrenuj model i zapisz go do folderu aplikacji.")
        return
    
    # Strona główna
    if page == "Strona główna":
        show_home_page()
    
    # Klasyfikator
    elif page == "Klasyfikator":
        show_classifier_page(model, vectorizer)
    
    # Analiza modelu
    elif page == "Analiza modelu":
        show_analysis_page(metrics)
    
    # Przykłady
    elif page == "Przykłady":
        show_examples_page(model, vectorizer)


def show_home_page():
    """Wyświetla stronę główną."""
    st.markdown('<p class="main-header">Klasyfikator SPAM/HAM</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Inteligentny system wykrywania niechcianych wiadomości</p>', unsafe_allow_html=True)
    
    # Opis projektu
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### O projekcie")
        st.markdown("""
        Ta aplikacja wykorzystuje uczenie maszynowe do automatycznej klasyfikacji 
        wiadomości tekstowych jako **SPAM** (niechciane) lub **HAM** (normalne).
        """)
        
        st.markdown("### Zbiór danych")
        st.markdown("""
        Aplikacja została wytrenowana na zbiorze danych, 
        który zawiera tysiące przykładów wiadomości oznaczonych jako spam lub ham.
        
        **Charakterystyka zbioru:**
        - 5625 wiadomości w języku angielskim
        - 3862 wiadomości HAM (68.66%) i 1763 wiadomości SPAM (31.34%)
        - Różnorodne przykłady spamu (loterie, oszustwa, reklamy) i normalnych wiadomości
        """)
    
    with col2:
        st.markdown("### Użyte techniki")
        st.markdown("""
        **Przetwarzanie tekstu:**
        - Tokenizacja i normalizacja tekstu
        - TF-IDF (Term Frequency-Inverse Document Frequency)
        
        **Model klasyfikacji:**
        - **LightGBM** - zaawansowany model gradient boosting
        - Wytrenowany z optymalnymi hiperparametrami
        - Wysoka dokładność i wydajność
        """)
        
        st.markdown("### Funkcjonalności")
        st.markdown("""
        - Klasyfikacja pojedynczych wiadomości
        - Wizualizacja prawdopodobieństw
        - Porównanie różnych modeli
        - Analiza metryk i macierzy pomyłek
        - Gotowe przykłady demonstracyjne
        """)
    
    # Stopka
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Projekt wykonany w ramach przedmiotu <b>Sztuczna Inteligencja</b></p>
    </div>
    """, unsafe_allow_html=True)


def show_classifier_page(model, vectorizer):
    """Wyświetla stronę klasyfikatora."""
    st.markdown("## Interaktywny Klasyfikator")
    st.markdown("*Model: **LightGBM** (z optymalnymi hiperparametrami)*")
    
    # Pole tekstowe
    message = st.text_area(
        "Wprowadź wiadomość do klasyfikacji:",
        height=150,
        placeholder="Wpisz tutaj treść wiadomości, którą chcesz sprawdzić..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        classify_button = st.button("🔍 Klasyfikuj", use_container_width=True, type="primary")
    
    if classify_button and message:
        # Wykonanie predykcji
        prediction, prob_ham, prob_spam = predict_message(message, model, vectorizer)
        
        st.markdown("---")
        st.markdown("### Wynik klasyfikacji")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="result-spam">
                    <h2 style="color: #c0392b; margin: 0;">🚫 SPAM</h2>
                    <p style="margin: 10px 0 0 0;">Ta wiadomość została sklasyfikowana jako spam.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-ham">
                    <h2 style="color: #27ae60; margin: 0;">✅ HAM</h2>
                    <p style="margin: 10px 0 0 0;">Ta wiadomość jest normalna (nie jest spamem).</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Prawdopodobieństwa")
            st.metric("HAM (Normalna)", f"{prob_ham*100:.1f}%")
            st.metric("SPAM", f"{prob_spam*100:.1f}%")
        
        with col2:
            # Wykres prawdopodobieństw
            fig = create_probability_chart(prob_ham, prob_spam)
            st.plotly_chart(fig, use_container_width=True)
    
    elif classify_button and not message:
        st.warning("Proszę wprowadzić wiadomość do klasyfikacji.")


def show_analysis_page(metrics):
    """Wyświetla stronę analizy modelu."""
    st.markdown("## Analiza modelu")
    
    # Tabela porównawcza wszystkich modeli
    st.markdown("### Porównanie wszystkich modeli")
    
    comparison_data = []
    for model_name in ['Multinomial Naive Bayes', 'Logistic Regression', 'LightGBM', 'LightGBM (stuningowany)']:
        model_metrics = metrics.get(model_name, {})
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{model_metrics.get('accuracy', 0)*100:.2f}%",
            'Precision': f"{model_metrics.get('precision', 0)*100:.2f}%",
            'Recall': f"{model_metrics.get('recall', 0)*100:.2f}%",
            'F1-Score': f"{model_metrics.get('f1_score', 0)*100:.2f}%"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Wykres porównawczy F1-Score
    st.markdown("### Porównanie F1-Score")
    fig_comparison = create_metrics_comparison_chart(metrics)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Macierze pomyłek dla wszystkich modeli
    st.markdown("### Macierze pomyłek")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Multinomial Naive Bayes")
        model_metrics = metrics.get('Multinomial Naive Bayes', {})
        cm = model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm = create_confusion_matrix_plot(np.array(cm), "Multinomial Naive Bayes")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("#### LightGBM (domyślne)")
        model_metrics = metrics.get('LightGBM', {})
        cm = model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm = create_confusion_matrix_plot(np.array(cm), "LightGBM (domyślne)")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("#### Logistic Regression")
        model_metrics = metrics.get('Logistic Regression', {})
        cm = model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm = create_confusion_matrix_plot(np.array(cm), "Logistic Regression")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("#### LightGBM (stuningowany) ⭐")
        model_metrics = metrics.get('LightGBM (stuningowany)', {})
        cm = model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm = create_confusion_matrix_plot(np.array(cm), "LightGBM (stuningowany)")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("""
    **Interpretacja macierzy pomyłek:**
    - **TN (True Negative)**: Poprawnie sklasyfikowane HAM
    - **FP (False Positive)**: HAM błędnie sklasyfikowane jako SPAM
    - **FN (False Negative)**: SPAM błędnie sklasyfikowany jako HAM
    - **TP (True Positive)**: Poprawnie sklasyfikowane SPAM
    """)
    
    st.markdown("---")
    
    # Metryki szczegółowe dla najlepszego modelu
    st.markdown("### Szczegółowe metryki najlepszego modelu: **LightGBM (stuningowany)**")
    
    model_metrics = metrics.get('LightGBM (stuningowany)', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{model_metrics.get('accuracy', 0)*100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Precision",
            value=f"{model_metrics.get('precision', 0)*100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Recall",
            value=f"{model_metrics.get('recall', 0)*100:.2f}%"
        )
    
    with col4:
        st.metric(
            label="⚖️ F1-Score",
            value=f"{model_metrics.get('f1_score', 0)*100:.2f}%"
        )
    
    # Analiza porównawcza
    st.markdown("---")
    st.markdown("### Wnioski z porównania")
    
    st.markdown("""
    **Najlepszy model: LightGBM (Tuned) ⭐**
    
    Model LightGBM z optymalnymi hiperparametrami (po tuningu RandomizedSearchCV) osiągnął najlepsze wyniki:
    - **Najwyższy F1-Score (98.22%)** - doskonała równowaga między precision i recall
    - **Najwyższa Accuracy (98.22%)** - najlepsza ogólna dokładność
    - **Najmniej błędów**: tylko 7 false positives i 13 false negatives
    
    **Porównanie modeli:**
    
    1. **Multinomial Naive Bayes** (F1: 93.51%):
       - Bardzo wysoka precision (99.68%), ale najniższy recall (88.07%)
       - Konserwatywny - rzadko oznacza jako spam, ale prawie zawsze ma rację
       - 42 false negatives (dużo spamu przechodzi)
    
    2. **Logistic Regression** (F1: 93.91%):
       - Zrównoważone wyniki, lepsza niż Naive Bayes
       - Recall 89.77% - nadal dużo pominiętego spamu (36 FN)
    
    3. **LightGBM domyślne** (F1: 96.34%):
       - Znacząca poprawa: recall 97.16%
       - Tylko 10 false negatives
       - Już lepszy od modeli klasycznych
    
    4. **LightGBM (Tuned)** (F1: 98.22%):
       - **Najlepszy w każdej metryce**
       - Tuning poprawił F1-Score o 1.88 p.p. względem domyślnego LightGBM
       - Tylko 7 FP (0.91% ham błędnie jako spam)
       - Tylko 13 FN (3.69% spam błędnie jako ham)
    
    **Wpływ tuningu hiperparametrów:**
    
    Optymalizacja hiperparametrów za pomocą RandomizedSearchCV (60 iteracji, 5-fold CV) przyniosła znaczącą poprawę:
    - Accuracy: +0.53 p.p. (97.69% → 98.22%)
    - F1-Score: +1.88 p.p. (96.34% → 98.22%)
    - Precision: +2.45 p.p. (95.53% → 97.98%)
    - Zmniejszenie FP o 56% (16 → 7)
    
    **Wybór modelu:**
    
    W aplikacji wykorzystywany jest **LightGBM (Tuned)** ze względu na:
    1. Najwyższą skuteczność we wszystkich metrykach
    2. Bardzo niską liczbę false positives (tylko 7) - ważne wiadomości nie są blokowane
    3. Skuteczne wykrywanie spamu (96.31% recall)
    4. Potwierdzenie skuteczności tuningu hiperparametrów
    """)
    
    # Wyjaśnienia metryk
    with st.expander("Wyjaśnienie metryk"):
        st.markdown("""
        - **Accuracy** - Ogólna dokładność modelu (ile procent predykcji jest poprawnych)
        - **Precision** - Precyzja dla klasy SPAM (jaki procent wiadomości oznaczonych jako SPAM faktycznie jest spamem)
        - **Recall** - Czułość dla klasy SPAM (jaki procent rzeczywistych spamów został wykryty)
        - **F1-Score** - Średnia harmoniczna precision i recall (równowaga między nimi)
        
        **W kontekście filtrowania spamu:**
        - Wysoka **Precision** oznacza mało false positives (ważne wiadomości nie są błędnie oznaczane jako spam)
        - Wysoki **Recall** oznacza mało false negatives (spam jest skutecznie wykrywany)
        - **F1-Score** łączy obie metryki w jedną wartość
        """)


def show_examples_page(model, vectorizer):
    """Wyświetla stronę z przykładami."""
    st.markdown("## Przykłady demonstracyjne")
    st.markdown("*Model: **LightGBM** (z optymalnymi hiperparametrami)*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Przykłady SPAM")
        for i, example in enumerate(EXAMPLE_SPAM):
            with st.expander(f"Spam #{i+1}"):
                st.markdown(f"*{example}*")
                if st.button(f"Klasyfikuj", key=f"spam_{i}"):
                    prediction, prob_ham, prob_spam = predict_message(example, model, vectorizer)
                    
                    result = "🚫 SPAM" if prediction == 1 else "✅ HAM"
                    color = "#e74c3c" if prediction == 1 else "#27ae60"
                    
                    st.markdown(f"**Wynik:** <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
                    st.progress(prob_spam, text=f"SPAM: {prob_spam*100:.1f}%")
    
    with col2:
        st.markdown("### Przykłady HAM")
        for i, example in enumerate(EXAMPLE_HAM):
            with st.expander(f"Ham #{i+1}"):
                st.markdown(f"*{example}*")
                if st.button(f"Klasyfikuj", key=f"ham_{i}"):
                    prediction, prob_ham, prob_spam = predict_message(example, model, vectorizer)
                    
                    result = "🚫 SPAM" if prediction == 1 else "✅ HAM"
                    color = "#e74c3c" if prediction == 1 else "#27ae60"
                    
                    st.markdown(f"**Wynik:** <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
                    st.progress(prob_ham, text=f"HAM: {prob_ham*100:.1f}%")

def preprocessing(text):
    """
    Pipeline przetwarzania tekstu:
    1. Konwersja do małych liter
    2. Usunięcie znaków specjalnych i cyfr
    3. Usunięcie nadmiarowych spacji
    4. Tokenizacja + usunięcie stop words + lematyzacja
    5. Połączenie tokenów w jeden string (dla klasyfikacji tekstu)
    """
    nlp = spacy.load('en_core_web_sm')
    text = text.lower()
    text = nlp(text)
    tokens = [token for token in text if token.is_alpha]
    tokens = [token.lemma_ for token in tokens]
    tokens = [ token for token in tokens if token not in STOP_WORDS]
   
    return ' '.join(tokens)

if __name__ == "__main__":
    main()
