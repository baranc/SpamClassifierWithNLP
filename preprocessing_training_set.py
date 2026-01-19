#!/usr/bin/env python
# coding: utf-8

# In[144]:


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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
import os


# In[3]:


data_path = os.path.join(os.path.dirname(__file__), "spam_NLP.csv")
df = pd.read_csv(data_path)


# In[69]:


print("=== 1. Sprawdzenie brakujących danych (missing values) ===")
print("Liczba brakujących wartości w każdej kolumnie:")
print(df.isnull().sum())
print("\nCałkowita liczba brakujących wartości w zbiorze:", df.isnull().sum().sum())

print("\n=== 2. Sprawdzenie duplikatów ===")
# Pełne duplikaty wierszy (identyczne CATEGORY + MESSAGE)
full_duplicates = df.duplicated().sum()
print(f"Liczba pełnych duplikatów wierszy: {full_duplicates}")

# Duplikaty tylko w kolumnie MESSAGE (niezależnie od kategorii)
message_duplicates = df['MESSAGE'].duplicated().sum()
print(f"Liczba duplikatów samych wiadomości (kolumna MESSAGE): {message_duplicates}")
print(f"\nLiczba wierszy przed usunięciem duplikatów: {len(df)}")

print("\n=== 3. Usunięcie duplikatów ===")
# Usunięcie pełnych duplikatów wierszy
df = df.drop_duplicates()

print(f"Liczba wierszy po usunięciu duplikatów: {len(df)}")


# In[3]:


print("Liczba wszystkich wiadomości:", len(df))
print("\n=== 1. Liczba próbek w każdej klasie ===")
print(df['CATEGORY'].value_counts().sort_index())


# In[62]:


hamSpamShare = df['CATEGORY'].value_counts().sort_index()/len(df)
hamSpamShare


# In[5]:


df['length'] = df['MESSAGE'].str.len()

print("\n=== 2. Rozkład długości wiadomości ===")
print(df['length'].describe())


# In[57]:


print("Ogólna mediana długości:", df['length'].median())

# Statystyki dla HAM (kategoria 0)
ham = df[df['CATEGORY'] == 0]
print("\nHAM (0) średnia długości:", ham['length'].describe())
print("HAM (0) mediana długości:", ham['length'].median())

# Statystyki dla SPAM (kategoria 1)
spam = df[df['CATEGORY'] == 1]
print("\nSPAM (1) średnia długości:", spam['length'].describe())
print("SPAM (1) mediana długości:", spam['length'].median())


# In[55]:


print("\n=== 3. Przykłady wiadomości ===")

print("\nPrzykładowe wiadomości SPAM (CATEGORY = 1):")
spam_examples = df[df['CATEGORY'] == 1].sample(n=3, random_state=20)  # 3 losowe przykłady
for i, row in spam_examples.iterrows():
    print(f"\n--- Przykład {i} (długość: {row['length']} znaków) ---")
    print(row['MESSAGE'][:500] + "..." if len(row['MESSAGE']) > 500 else row['MESSAGE'])

print("\nPrzykładowe wiadomości HAM (CATEGORY = 0):")
ham_examples = df[df['CATEGORY'] == 0].sample(n=3, random_state=20)
for i, row in ham_examples.iterrows():
    print(f"\n--- Przykład {i} (długość: {row['length']} znaków) ---")
    print(row['MESSAGE'][:500] + "..." if len(row['MESSAGE']) > 500 else row['MESSAGE'])


# In[54]:


# === 1. Wykres słupkowy rozkładu klas ===
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='CATEGORY', hue='CATEGORY', legend=False, palette={0: 'blue', 1: 'red'})
plt.title('Rozkład klas (0 = ham, 1 = spam)')
plt.xlabel('Klasa')
plt.ylabel('Liczba wiadomości')
plt.xticks([0, 1], ['Ham (0)', 'Spam (1)'])
for i, count in enumerate(df['CATEGORY'].value_counts().sort_index()):
    plt.text(i, count + 10, str(count), ha='center', fontsize=12)
plt.tight_layout()
try:
    plt.savefig("rozklad_klas.pdf", bbox_inches="tight", pad_inches=0.04, transparent=False)
    print("Wykres 'rozklad_klas.pdf' został zapisany.")
except Exception as e:
    print(f"Błąd podczas zapisywania wykresu 'rozklad_klas.pdf': {e}")
plt.show()


# In[39]:


x_cut = df['length'].quantile(0.96)
df_cut = df[df['length'] <= x_cut]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.histplot(
    data=df_cut[df_cut['CATEGORY'] == 0],
    x='length',
    bins=30,
    ax=axes[0],
    color='tab:blue'
)
axes[0].set_title('Ham')
axes[0].set_xlabel('Długość wiadomości')
axes[0].set_ylabel('Liczba wiadomości')

sns.histplot(
    data=df_cut[df_cut['CATEGORY'] == 1],
    x='length',
    bins=30,
    ax=axes[1],
    color='tab:red'
)
axes[1].set_title('Spam')
axes[1].set_xlabel('Długość wiadomości')

plt.tight_layout()
try:
    plt.savefig("histogram.pdf", bbox_inches="tight", pad_inches=0.04, transparent=False)
    print("Wykres 'histogram.pdf' został zapisany.")
except Exception as e:
    print(f"Błąd podczas zapisywania wykresu 'histogram.pdf': {e}")
plt.show()


# In[45]:


# === 3. Chmura słów dla spam i ham ===
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
spam_text = ' '.join(df[df['CATEGORY'] == 1]['MESSAGE'].astype(str))
ham_text  = ' '.join(df[df['CATEGORY'] == 0]['MESSAGE'].astype(str))

# Ham
wordcloud_ham = WordCloud(width=800, height=400, background_color='white',
                          colormap='Blues', max_words=200).generate(ham_text)
axs[0].imshow(wordcloud_ham, interpolation='bilinear')
axs[0].set_title('Chmura słów – HAM', fontsize=16)
axs[0].axis('off')
# Spam
wordcloud_spam = WordCloud(width=800, height=400, background_color='white',
                           colormap='Reds', max_words=200).generate(spam_text)
axs[1].imshow(wordcloud_spam, interpolation='bilinear')
axs[1].set_title('Chmura słów – SPAM', fontsize=16)
axs[1].axis('off')

plt.tight_layout()
try:
    plt.savefig("chmura_slow.pdf", bbox_inches="tight", pad_inches=0.04, transparent=False)
    print("Wykres 'chmura_slow.pdf' został zapisany.")
except Exception as e:
    print(f"Błąd podczas zapisywania wykresu 'chmura_slow.pdf': {e}")
plt.show()


# In[9]:


nlp = spacy.load('en_core_web_sm')


# In[4]:


def preprocessing(text):
    """
    Pipeline przetwarzania tekstu:
    1. Konwersja do małych liter
    2. Usunięcie znaków specjalnych i cyfr
    3. Usunięcie nadmiarowych spacji
    4. Tokenizacja + usunięcie stop words + lematyzacja
    5. Połączenie tokenów w jeden string (dla klasyfikacji tekstu)
    """
    text = text.lower()
    text = nlp(text)
    tokens = [token for token in text if token.is_alpha]
    tokens = [token.lemma_ for token in tokens]
    tokens = [ token for token in tokens if token not in STOP_WORDS]

    return ' '.join(tokens)


# In[10]:


# Test funkcji na jednej wiadomości
print("Przykład przed:")
print(df['MESSAGE'].iat[0][:500])
print("\nPrzykład po:")
print(preprocessing(df['MESSAGE'].iat[0])[:500])


# In[11]:


# Zastosowanie pipeline do całego zbioru
print("\nPrzetwarzanie całego zbioru (może chwilę potrwać)...")
df['clean_message'] = df['MESSAGE'].apply(preprocessing)


# In[13]:


# Podgląd wyników
print(df[['CATEGORY', 'MESSAGE', 'clean_message']].head())


# In[22]:


vec = CountVectorizer()
X = vec.fit_transform(df['clean_message'])
y = df['CATEGORY']


# In[18]:


X


# In[23]:


import numpy as np

feature_names = vec.get_feature_names_out()

print("=== Analiza wektoryzacji (Bag of Words) ===\n")

# 1. Wymiarowość przestrzeni cech
num_features = X.shape[1]
print(f"1. Wymiarowość (liczba unikalnych cech/słów): {num_features}")

# 2. Gęstość macierzy (sparsity)
total_elements = X.shape[0] * X.shape[1]
non_zero_elements = X.nnz
density_percent = (non_zero_elements / total_elements) * 100
sparsity_percent = 100 - density_percent

print(f"2. Liczba niezerowych elementów: {non_zero_elements}")
print(f"   Gęstość macierzy: {density_percent:.6f}%")
print(f"   Sparsity (rzadkość): {sparsity_percent:.6f}%\n")

# 3. Najważniejsze słowa dla każdej klasy (top 20)
top_n = 20

for class_label in [0, 1]:
    class_name = "HAM (0)" if class_label == 0 else "SPAM (1)"
    print(f"3. Top {top_n} najważniejszych słów dla klasy {class_name}:")

    # Indeksy wierszy danej klasy
    class_indices = np.where(y == class_label)[0]

    # Suma wystąpień cech w danej klasie
    class_sum = X[class_indices].sum(axis=0)  # macierz 1 x num_features
    class_sum = np.array(class_sum).squeeze()     # konwersja na tablicę 1D

    # Indeksy posortowane malejąco według sumy
    top_indices = np.argsort(class_sum)[-top_n:][::-1]

    # Wyświetlenie słów i ich sumarycznych wystąpień
    for rank, idx in enumerate(top_indices, 1):
        word = feature_names[idx]
        count = int(class_sum[idx])
        print(f"   {rank:2}. {word:<15} : {count:,} wystąpień")

    print()


# In[52]:


joblib.dump(vec, 'BofW_vectorizer.pkl')


# In[24]:


df.to_csv('spam_NLP_processed.csv', index=False)