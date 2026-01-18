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


# In[159]:


vec1 = TfidfVectorizer()
X_tfidf = vec1.fit_transform(df['clean_message'])
y = df['CATEGORY']


# In[39]:


feature_names = vec1.get_feature_names_out()

print("=== Analiza wektoryzacji (TF-IDF) ===\n")

# 1. Wymiarowość przestrzeni cech
num_features = X_tfidf.shape[1]
print(f"1. Wymiarowość (liczba unikalnych cech/słów): {num_features}")

# 2. Gęstość macierzy (sparsity)
total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
non_zero_elements = X_tfidf.nnz
density_percent = (non_zero_elements / total_elements) * 100
sparsity_percent = 100 - density_percent

print(f"2. Liczba niezerowych elementów: {non_zero_elements}")
print(f"   Gęstość macierzy: {density_percent:.6f}%")
print(f"   Sparsity (rzadkość): {sparsity_percent:.6f}%\n")

# 3. Najważniejsze słowa dla każdej klasy (top 20)
top_n = 20

for class_label in [0, 1]:
    class_name = "HAM (0)" if class_label == 0 else "SPAM (1)"
    print(f"3. Top {top_n} najważniejszych słów dla klasy {class_name} (wg sumy TF-IDF):")

    # Indeksy wierszy danej klasy
    class_indices = np.where(y == class_label)[0]

    # Suma wartości TF-IDF w danej klasie
    class_sum = X_tfidf[class_indices].sum(axis=0)  # macierz 1 x num_features
    class_sum = np.array(class_sum).squeeze()      # konwersja na tablicę 1D

    # Indeksy posortowane malejąco według sumy TF-IDF
    top_indices = np.argsort(class_sum)[-top_n:][::-1]

    # Wyświetlenie słów i ich sumarycznych wartości TF-IDF
    for rank, idx in enumerate(top_indices, 1):
        word = feature_names[idx]
        tfidf_score = class_sum[idx]
        print(f"   {rank:2}. {word:<20} : {tfidf_score:,.2f}")

    print()


# In[160]:


joblib.dump(vec1, 'tfidf_vectorizer.pkl')


# In[40]:


# === Podział danych: 80% train, 20% test ===
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,               
    y,                     
    test_size=0.2,         
    train_size=0.8,        
    random_state=20,       
    stratify=y             
)

print(f"\nPodział danych zakończony:")
print(f"Train: {X_train.shape[0]} wiadomości ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} wiadomości ({X_test.shape[0]/len(df)*100:.1f}%)")

print("\nRozkład klas w zbiorze treningowym:")
print(pd.Series(y_train).value_counts(normalize=True).round(3))

print("\nRozkład klas w zbiorze testowym:")
print(pd.Series(y_test).value_counts(normalize=True).round(3))


# In[49]:


cross_val_score(X=X_train, y=y_train, estimator=MultinomialNB())


# In[55]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# In[108]:


model = MultinomialNB()
model.fit(X=X_train, y=y_train)
y_predict = model.predict(X=X_test)


# In[109]:


# === Ewaluacja modelu ===
print("\n=== Wyniki na zbiorze testowym ===")
print(f"Accuracy: {accuracy_score(y_test, y_predict):.4f}")


# In[99]:


cr = classification_report(y_true = y_test, y_pred = y_predict, digits=4)


# In[59]:


cm = confusion_matrix(y_true = y_test, y_pred = y_predict)


# In[100]:


print(cr)


# In[61]:


print(cm)


# In[63]:


ConfusionMatrixDisplay(confusion_matrix=cm).plot()


# In[66]:


joblib.dump(model, 'multinomial_nb_model.pkl')


# In[79]:


cross_val_score(X=X_train, y=y_train, estimator=LogisticRegression(random_state=20))


# In[110]:


model = LogisticRegression(random_state=20)
model.fit(X=X_train, y=y_train)
y_predict = model.predict(X=X_test)


# In[111]:


# === Ewaluacja modelu ===
print("\n=== Wyniki na zbiorze testowym ===")
print(f"Accuracy: {accuracy_score(y_test, y_predict):.4f}")
cr = classification_report(y_true = y_test, y_pred = y_predict, digits=4)
print(cr)
cm = confusion_matrix(y_true = y_test, y_pred = y_predict)
print(cm)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()


# In[72]:


joblib.dump(model, 'logistic_regression_model.pkl')


# In[78]:


cross_val_score(X=X_train, y=y_train, estimator=LGBMClassifier(random_state=20))


# In[106]:


model = LGBMClassifier(random_state=20)
model.fit(X=X_train, y=y_train)
y_predict = model.predict(X=X_test)


# In[141]:


### === Ewaluacja modelu ===


# In[107]:


# === Ewaluacja modelu ===
print("\n=== Wyniki na zbiorze testowym ===")
print(f"Accuracy: {accuracy_score(y_test, y_predict):.4f}")
cr = classification_report(y_true = y_test, y_pred = y_predict, digits=4)
print(cr)
cm = confusion_matrix(y_true = y_test, y_pred = y_predict)
print(cm)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()


# In[84]:


joblib.dump(model, 'LightGBM_model.pkl')


# In[142]:


# === Tuning hiperparametrów Logistic Regression ===
print("=== GridSearchCV – tuning hiperparametrów Logistic Regression ===\n")

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],           
    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg', 'newton-cholesky', 'sag'],
    'max_iter': [1000]                            
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)

grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=20),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,                  
    verbose=2,                  
    refit=True                  
)

grid_search.fit(X_train, y_train)

# === Wyniki tuningu ===
print("\n=== Najlepsze wyniki po GridSearchCV ===")
print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepszy średni F1-weighted (CV): {grid_search.best_score_:.4f}")

# Szczegółowe wyniki (top 5 kombinacji)
cv_results = pd.DataFrame(grid_search.cv_results_)
top_results = cv_results.sort_values('mean_test_score', ascending=False).head(5)
print("\nTop 5 kombinacji parametrów:")
print(top_results[['params', 'mean_test_score', 'std_test_score']].round(4))

# === Najlepszy model ===
best_model = grid_search.best_estimator_

# === Ewaluacja na hold-out teście (jeśli użyłeś podziału) ===
print("\n=== Ewaluacja najlepszego modelu na hold-out teście ===")
y_pred = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['HAM (0)', 'SPAM (1)'], digits=4))


# In[96]:


top_results['params'][70]


# In[112]:


spamExample = """Szanowny Użytkowniku,

Z radością informujemy, że Twój adres e-mail został **wylosowany** w naszej międzynarodowej loterii online!

🏆 **Nagroda: 50 000 PLN**
⏰ Oferta ważna tylko przez **24 GODZINY**

Aby odebrać nagrodę, kliknij poniższy link i potwierdź swoje dane:
👉 [http://secure-prize-confirmation.example/login](http://secure-prize-confirmation.example/login)

W celu weryfikacji prosimy o podanie:

* imienia i nazwiska
* numeru karty płatniczej
* daty ważności karty

Brak potwierdzenia spowoduje **utratę nagrody**.

Z wyrazami szacunku,
Dział Wygranych Online
Global Rewards Center
[contact@global-rewards.example](mailto:contact@global-rewards.example)"""


# In[120]:


spamEx = preprocessing(spamExample)
spamEx
spamEx = vec1.transform([spamEx])


# In[151]:


y_predict = best_lgbm.predict(spamEx)


# In[152]:


print(y_predict)


# In[124]:


spamEx1 = """Dear Customer,

We have detected **unusual activity** on your account. For your protection, your access will be **temporarily suspended** unless you verify your information immediately.

✅ Verify your account within **12 HOURS** to avoid permanent closure:
👉 [http://secure-account-verification.example/login](http://secure-account-verification.example/login)

To complete verification, please confirm:

* Full name
* Date of birth
* Credit card number for identity validation

Failure to act may result in **loss of access and data**.

Thank you for your prompt cooperation,
Security Team
Online Account Services
[support@account-services.example](mailto:support@account-services.example)
"""
spamEx1 = preprocessing(spamEx1)
spamEx1 = vec1.transform([spamEx1])


# In[153]:


y_predict = best_lgbm.predict(spamEx1)
print(y_predict)


# In[132]:


hamEx = """Dear John,

Thank you for taking the time to meet with me today. I appreciate the opportunity to discuss our current progress and future plans.

As agreed, I will send you the updated documentation by Friday. Please let me know if you would like me to include any additional details or make adjustments before then.

If you have any questions in the meantime, feel free to contact me.

Best regards,
Bill O’Farrell
"""
hamEx = preprocessing(hamEx)
hamEx = vec1.transform([hamEx])


# In[154]:


y_predict = best_lgbm.predict(hamEx)
print(y_predict)


# In[130]:


hamEx1 = """Hi Sarah,

I wanted to provide a quick update on the current status of the project.

The main features have been implemented and are now in the testing phase. So far, everything is on track, and we don’t expect any delays. I’ll keep you informed if anything changes.

Please let me know if you have any questions or need additional information.

Kind regards,
Bill O’Farrell
"""
hamEx1 = preprocessing(hamEx1)
hamEx1 = vec1.transform([hamEx1])


# In[155]:


y_predict = best_lgbm.predict(hamEx1)
print(y_predict)


# In[136]:


hamEx2 = """
Dear Customer,

Thank you for contacting us. We would like to inform you that your request is currently being reviewed by our support team.

At this stage, no additional information is required from you. We will get back to you as soon as the analysis is complete or if further details are needed.

We appreciate your patience and thank you for choosing our services.

Kind regards,
Bill O’Farrell
Customer Support Team
"""
hamEx2 = preprocessing(hamEx2)
print(hamEx2)
hamEx2 = vec1.transform([hamEx2])


# In[156]:


y_predict = best_lgbm.predict(hamEx2)
print(y_predict)


# In[137]:


hamEx3 = """Dear IT Support Team,

I am experiencing an issue with accessing the internal application. Since this morning, the system fails to load after login and displays an unexpected error message.

I have already tried restarting the application and clearing the cache, but the issue persists.

Please let me know if you need any additional information from my side.

Kind regards,
Bill O’Farrell
"""
hamEx3 = preprocessing(hamEx3)
print(hamEx3)
hamEx3 = vec1.transform([hamEx3])


# In[157]:


y_predict = best_lgbm.predict(hamEx3)
print(y_predict)


# In[145]:


# =====================================
# 2. Definicja rozkładów parametrów
# =====================================

param_distributions = {
    'n_estimators':     randint(100, 1000),          # liczba drzew
    'learning_rate':    uniform(0.01, 0.29),         # 0.01 – 0.30
    'max_depth':        randint(3, 15),              # -1 = brak limitu
    'num_leaves':       randint(20, 150),            # liczba liści w drzewie
    'min_child_samples': randint(20, 100),           # minimalna liczba próbek w liściu
    'subsample':        uniform(0.6, 0.4),           # 0.6–1.0
    'colsample_bytree': uniform(0.6, 0.4),           # 0.6–1.0
    'reg_alpha':        uniform(0.0, 1.0),           # L1 regularization
    'reg_lambda':       uniform(0.0, 1.0),           # L2 regularization
}


# In[146]:


# =====================================
# 3. RandomizedSearchCV
# =====================================

print("=== RandomizedSearchCV – tuning LightGBM ===\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)

random_search = RandomizedSearchCV(
    estimator=LGBMClassifier(
        random_state=20,
        n_jobs=-1,
        verbose=-1
    ),
    param_distributions=param_distributions,
    n_iter=60,                      # ile losowych kombinacji przetestować
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=20,
    verbose=1,
    refit=True
)


# In[147]:


# Uruchomienie tuningu
random_search.fit(X_train, y_train)


# In[148]:


# =====================================
# 4. Wyniki
# =====================================

print("\n=== Najlepsze wyniki po RandomizedSearchCV ===")
print(f"Najlepsze parametry:\n{random_search.best_params_}")
print(f"Najlepszy średni F1-weighted (CV): {random_search.best_score_:.4f}\n")

# Top 5 kombinacji
cv_results = pd.DataFrame(random_search.cv_results_)
top5 = cv_results.sort_values('mean_test_score', ascending=False).head(5)
print("Top 5 najlepszych kombinacji:")
print(top5[['params', 'mean_test_score', 'std_test_score']].round(4))

# Najlepszy model
best_lgbm = random_search.best_estimator_


# In[149]:


# =====================================
# 5. Finalna ewaluacja na hold-out teście
# =====================================

print("\n=== Ewaluacja najlepszego modelu LightGBM na zbiorze testowym ===")
y_pred = best_lgbm.predict(X_test)

print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-weighted:   {f1_score(y_test, y_pred, average='weighted'):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['HAM (0)', 'SPAM (1)'], digits=4))


# In[150]:


# =====================================
# 6. Zapis modelu
# =====================================

joblib.dump(best_lgbm, 'best_lightgbm_tuned.pkl')


# In[ ]:




