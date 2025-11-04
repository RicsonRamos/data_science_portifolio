# %% [markdown]
# **Configuração do ambiente e carregamento das ferramentas (Pandas, NLTK, sklearn) e dos dados.**

# %% 
import re
import nltk
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from pathlib import Path


# %% 
# Carregando os dados
df = pd.read_csv('data/raw/airlines_reviews.csv')

# %% 
# Verificando e baixando dependências do NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# %% 
# Assumindo que seu DataFrame de reviews limpos se chama 'df'
df.head(1)

# %% 
# Verificando as colunas e tipos de dados
df.info()

# %% [markdown]
# **Convertendo a coluna 'Review Date' para o formato datetime**

# %% 
df['Review Date'] = pd.to_datetime(df['Review Date'])

# %% [markdown]
# **Limpeza de ruído e padronização da coluna 'Verified'.**

# %% 
df['Verified'] = df['Verified'].replace({'*Unverified*': 'False', 'NotVerified': 'False'})
df = df[df['Verified'].isin(['True', 'False'])]
df['Verified'] = df['Verified'].map({'True': 1, 'False': 0}).astype(bool)

# %% [markdown]
# **Transformando strings de data ('Month Flown') em objetos datetime.**

# %% 
df['Month Flown'] = pd.to_datetime(df['Month Flown'], format='%B %Y')
df['Year Flown'] = df['Month Flown'].dt.year
df['Month Flown'] = df['Month Flown'].dt.month

# %% [markdown]
# **Conversão da coluna 'Recommended' para 1/0.**

# %% 
df['Recommended'] = df['Recommended'].map({'Yes': 1, 'No': 0}).astype(bool)

# %% [markdown]
# **Padronizando as colunas do tipo 'object' para string.**

# %% 
df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).astype('string')

# %% 
df.head()

# %% [markdown]
# **Verificando os valores das colunas numéricas.**

# %% 
df_int = df.select_dtypes(include=['int64'])
df_int.describe()

# %% [markdown]
# **Padronizando a coluna 'Inflight Entertainment'.**

# %% 
df['Inflight Entertainment'] = df['Inflight Entertainment'].replace(0, 1)

# %% [markdown]
# **Padronizando as colunas de rating.**

# %% 
numeric_cols = df_int.columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df_int)

# %% [markdown]
# **Configuração e Inicialização do VADER.**

# %% 
sia = SentimentIntensityAnalyzer()

# %% [markdown]
# **Função de Classificação do Sentimento.**

def classificar_sentimento(texto):
    if not isinstance(texto, str):
        return 'Neutro'
    score = sia.polarity_scores(texto).get('compound', 0)
    return 'Positivo' if score >= 0.05 else 'Negativo' if score <= -0.05 else 'Neutro'

# %% [markdown]
# **Aplicação da Análise de Sentimento.**

if 'Reviews' in df.columns:
    df['Sentimento'] = df['Reviews'].apply(classificar_sentimento)
    print("Distribuição de Sentimentos (VADER)")
    print(df['Sentimento'].value_counts())
else:
    print("ERRO: Coluna 'Reviews' não encontrada no DataFrame 'df'.")

# %% [markdown]
# **CountVectorizer e LDA para modelagem de tópicos.**

vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=5)
dtm = vectorizer.fit_transform(df['Reviews'])

lda = LatentDirichletAllocation(n_components=5, random_state=42, learning_method='batch')
lda.fit(dtm)

# %% [markdown]
# **Atribuição do Tópico Mais Provável.**

topic_probabilities = lda.transform(dtm)
df['Topico_ID'] = topic_probabilities.argmax(axis=1)

topic_mapping = {
    0: 'Atendimento e Ticketing',
    1: 'Experiência em Voo (Assento/Comida)',
    2: 'Qatar/Doha (Hub e Serviço)',
    3: 'Problemas em Solo (Atrasos/Aeroporto)',
    4: 'Rotas/Aéreas Específicas (AF/EVA)'
}
df['Topico_Principal'] = df['Topico_ID'].map(topic_mapping)

# %% [markdown]
# **Criação da Tabela de Contingência (Sentimento vs. Tópico).**

contingency_table = pd.crosstab(df['Topico_Principal'], df['Sentimento'])
topic_sentiment_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

print("\n--- Tabela de Análise Prescritiva: % Sentimento por Tópico ---")
analysis_final = topic_sentiment_pct[['Negativo']].sort_values(by='Negativo', ascending=False)
analysis_final['Total_Reviews'] = contingency_table.sum(axis=1)
print(analysis_final.to_markdown())

# %% [markdown]
# **Análise Quantitativa por Segmento.**

COLUNAS_SEGMENTACAO = ['Class', 'Type of Traveller', 'Airline']
TOPICO_ALVO = 'Experiência em Voo (Assento/Comida)'

print("\n--- Análise Quantitativa: % Negativo no Tópico 1 por Segmento ---")
df_topico_1 = df[df['Topico_Principal'] == TOPICO_ALVO]

for col in COLUNAS_SEGMENTACAO:
    if col in df_topico_1.columns:
        crosstab_seg = pd.crosstab(df_topico_1[col], df_topico_1['Sentimento'])
        segment_pct = crosstab_seg.div(crosstab_seg.sum(axis=1), axis=0) * 100
        print(f"\nResultados para a coluna: {col}")
        analise_segmento = segment_pct[['Negativo']].sort_values(by='Negativo', ascending=False)
        min_reviews = 50
        segmentos_relevantes = crosstab_seg.sum(axis=1)[crosstab_seg.sum(axis=1) >= min_reviews].index
        analise_segmento_relevante = analise_segmento.loc[analise_segmento.index.intersection(segmentos_relevantes)]
        print(analise_segmento_relevante.head(5).to_markdown(floatfmt=".2f"))
    else:
        print(f"Aviso: Coluna '{col}' não encontrada no DataFrame para segmentação.")

# %% [markdown]
# **Geração da Nuvem de Palavras.**

TOPICO_ALVO = 'Experiência em Voo (Assento/Comida)'
COMPANHIA_ALVO = 'Turkish Airlines'
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_wc(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\w\s]','', text, re.I|re.A).lower().strip()
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

textos_auditoria = df[(df['Sentimento'] == 'Negativo') & (df['Topico_Principal'] == TOPICO_ALVO) & (df['Airline'] == COMPANHIA_ALVO)]['Reviews']
texto_completo = ' '.join(textos_auditoria.apply(preprocess_text_wc))

wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='Reds_r', max_words=30).generate(texto_completo)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f'Foco Operacional: Reclamações Negativas - {COMPANHIA_ALVO} (Assento/Comida)', fontsize=16)
plt.show()

frequencias = wordcloud.words_.keys()
palavras_chave = list(frequencias)[:7]
print(f"\n--- 7 Palavras-Chave de Reclamação - {COMPANHIA_ALVO} ---")
print(palavras_chave)

# %% [markdown]
# **Implementação do Score de Risco Operacional (SRO).**

SRO_WATCH_LIST = {
    'Assento', 'Comida', 'Fila', 'Problema', 'Atraso', 'Serviço', 'Segurança', 'Reclamação', 'Cancelamento'
}

def calcular_sro(texto, lista_palavras):
    if not isinstance(texto, str):
        return 0
    texto = preprocess_text_wc(texto)
    return sum(1 for palavra in lista_palavras if palavra.lower() in texto)

df['SRO'] = df['Reviews'].apply(lambda x: calcular_sro(x, SRO_WATCH_LIST))
df['SRO'] = df['SRO'] / df['SRO'].max()  # Normalizando entre 0 e 1

# %% [markdown]
# **Salvando os Resultados**

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path('output')
out_dir.mkdir(exist_ok=True)

# Salvando análise quantitativa
analysis_final.to_csv(out_dir / f"analise_sentimento_{timestamp}.csv")

# Salvando visualizações
fig, ax = plt.subplots(figsize=(10, 6))
topic_sentiment_pct.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Distribuição de Sentimentos por Tópico')
plt.savefig(out_dir / f"sentimento_por_topico_{timestamp}.png")
plt.close(fig)

# %% 
# Fim do código.
# %% [markdown]
# **Configuração do ambiente e carregamento das ferramentas (Pandas, NLTK, sklearn) e dos dados.**

# %% 
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from pathlib import Path

# %% 
# Verificando e baixando dependências do NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# %% 
# Assumindo que seu DataFrame de reviews limpos se chama 'df'
df.head(1)

# %% 
# Verificando as colunas e tipos de dados
df.info()

# %% [markdown]
# **Convertendo a coluna 'Review Date' para o formato datetime**

# %% 
df['Review Date'] = pd.to_datetime(df['Review Date'])

# %% [markdown]
# **Limpeza de ruído e padronização da coluna 'Verified'.**

# %% 
df['Verified'] = df['Verified'].replace({'*Unverified*': 'False', 'NotVerified': 'False'})
df = df[df['Verified'].isin(['True', 'False'])]
df['Verified'] = df['Verified'].map({'True': 1, 'False': 0}).astype(bool)

# %% [markdown]
# **Transformando strings de data ('Month Flown') em objetos datetime.**

# %% 
df['Month Flown'] = pd.to_datetime(df['Month Flown'], format='%B %Y')
df['Year Flown'] = df['Month Flown'].dt.year
df['Month Flown'] = df['Month Flown'].dt.month

# %% [markdown]
# **Conversão da coluna 'Recommended' para 1/0.**

# %% 
df['Recommended'] = df['Recommended'].map({'Yes': 1, 'No': 0}).astype(bool)

# %% [markdown]
# **Padronizando as colunas do tipo 'object' para string.**

# %% 
df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).astype('string')

# %% 
df.head()

# %% [markdown]
# **Verificando os valores das colunas numéricas.**

# %% 
df_int = df.select_dtypes(include=['int64'])
df_int.describe()

# %% [markdown]
# **Padronizando a coluna 'Inflight Entertainment'.**

# %% 
df['Inflight Entertainment'] = df['Inflight Entertainment'].replace(0, 1)

# %% [markdown]
# **Padronizando as colunas de rating.**

# %% 
numeric_cols = df_int.columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df_int)

# %% [markdown]
# **Configuração e Inicialização do VADER.**

# %% 
sia = SentimentIntensityAnalyzer()

# %% [markdown]
# **Função de Classificação do Sentimento.**

def classificar_sentimento(texto):
    if not isinstance(texto, str):
        return 'Neutro'
    score = sia.polarity_scores(texto).get('compound', 0)
    return 'Positivo' if score >= 0.05 else 'Negativo' if score <= -0.05 else 'Neutro'

# %% [markdown]
# **Aplicação da Análise de Sentimento.**

if 'Reviews' in df.columns:
    df['Sentimento'] = df['Reviews'].apply(classificar_sentimento)
    print("Distribuição de Sentimentos (VADER)")
    print(df['Sentimento'].value_counts())
else:
    print("ERRO: Coluna 'Reviews' não encontrada no DataFrame 'df'.")

# %% [markdown]
# **CountVectorizer e LDA para modelagem de tópicos.**

vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=5)
dtm = vectorizer.fit_transform(df['Reviews'])

lda = LatentDirichletAllocation(n_components=5, random_state=42, learning_method='batch')
lda.fit(dtm)

# %% [markdown]
# **Atribuição do Tópico Mais Provável.**

topic_probabilities = lda.transform(dtm)
df['Topico_ID'] = topic_probabilities.argmax(axis=1)

topic_mapping = {
    0: 'Atendimento e Ticketing',
    1: 'Experiência em Voo (Assento/Comida)',
    2: 'Qatar/Doha (Hub e Serviço)',
    3: 'Problemas em Solo (Atrasos/Aeroporto)',
    4: 'Rotas/Aéreas Específicas (AF/EVA)'
}
df['Topico_Principal'] = df['Topico_ID'].map(topic_mapping)

# %% [markdown]
# **Criação da Tabela de Contingência (Sentimento vs. Tópico).**

contingency_table = pd.crosstab(df['Topico_Principal'], df['Sentimento'])
topic_sentiment_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

print("\n--- Tabela de Análise Prescritiva: % Sentimento por Tópico ---")
analysis_final = topic_sentiment_pct[['Negativo']].sort_values(by='Negativo', ascending=False)
analysis_final['Total_Reviews'] = contingency_table.sum(axis=1)
print(analysis_final.to_markdown())

# %% [markdown]
# **Análise Quantitativa por Segmento.**

COLUNAS_SEGMENTACAO = ['Class', 'Type of Traveller', 'Airline']
TOPICO_ALVO = 'Experiência em Voo (Assento/Comida)'

print("\n--- Análise Quantitativa: % Negativo no Tópico 1 por Segmento ---")
df_topico_1 = df[df['Topico_Principal'] == TOPICO_ALVO]

for col in COLUNAS_SEGMENTACAO:
    if col in df_topico_1.columns:
        crosstab_seg = pd.crosstab(df_topico_1[col], df_topico_1['Sentimento'])
        segment_pct = crosstab_seg.div(crosstab_seg.sum(axis=1), axis=0) * 100
        print(f"\nResultados para a coluna: {col}")
        analise_segmento = segment_pct[['Negativo']].sort_values(by='Negativo', ascending=False)
        min_reviews = 50
        segmentos_relevantes = crosstab_seg.sum(axis=1)[crosstab_seg.sum(axis=1) >= min_reviews].index
        analise_segmento_relevante = analise_segmento.loc[analise_segmento.index.intersection(segmentos_relevantes)]
        print(analise_segmento_relevante.head(5).to_markdown(floatfmt=".2f"))
    else:
        print(f"Aviso: Coluna '{col}' não encontrada no DataFrame para segmentação.")

# %% [markdown]
# **Geração da Nuvem de Palavras.**

TOPICO_ALVO = 'Experiência em Voo (Assento/Comida)'
COMPANHIA_ALVO = 'Turkish Airlines'
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_wc(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\w\s]','', text, re.I|re.A).lower().strip()
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

textos_auditoria = df[(df['Sentimento'] == 'Negativo') & (df['Topico_Principal'] == TOPICO_ALVO) & (df['Airline'] == COMPANHIA_ALVO)]['Reviews']
texto_completo = ' '.join(textos_auditoria.apply(preprocess_text_wc))

wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='Reds_r', max_words=30).generate(texto_completo)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f'Foco Operacional: Reclamações Negativas - {COMPANHIA_ALVO} (Assento/Comida)', fontsize=16)
plt.show()

frequencias = wordcloud.words_.keys()
palavras_chave = list(frequencias)[:7]
print(f"\n--- 7 Palavras-Chave de Reclamação - {COMPANHIA_ALVO} ---")
print(palavras_chave)

# %% [markdown]
# **Implementação do Score de Risco Operacional (SRO).**

SRO_WATCH_LIST = {
    'Assento', 'Comida', 'Fila', 'Problema', 'Atraso', 'Serviço', 'Segurança', 'Reclamação', 'Cancelamento'
}

def calcular_sro(texto, lista_palavras):
    if not isinstance(texto, str):
        return 0
    texto = preprocess_text_wc(texto)
    return sum(1 for palavra in lista_palavras if palavra.lower() in texto)

df['SRO'] = df['Reviews'].apply(lambda x: calcular_sro(x, SRO_WATCH_LIST))
df['SRO'] = df['SRO'] / df['SRO'].max()  # Normalizando entre 0 e 1

# %% [markdown]
# **Salvando os Resultados**

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path('output')
out_dir.mkdir(exist_ok=True)

# Salvando análise quantitativa
analysis_final.to_csv(out_dir / f"analise_sentimento_{timestamp}.csv")

# Salvando visualizações
fig, ax = plt.subplots(figsize=(10, 6))
topic_sentiment_pct.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Distribuição de Sentimentos por Tópico')
plt.savefig(out_dir / f"sentimento_por_topico_{timestamp}.png")
plt.close(fig)

# %% 
# Fim do código.
