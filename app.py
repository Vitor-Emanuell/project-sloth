import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Função para ler descrições de um arquivo .txt
def ler_descricoes(caminho_arquivo, rotulo):
    descricoes = []
    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha:
                descricoes.append({'descrição': linha, 'rótulo': rotulo})
    return descricoes

# Passo 1: Carregar e preparar os dados de NLP
descricoes_preguica = ler_descricoes('./sloth-recognizer/sloth-text-recognizer/bicho-preguica.txt', 'Bicho-preguiça')
descricoes_outros = ler_descricoes('./sloth-recognizer/sloth-text-recognizer/outros.txt', 'Outro animal')

# Combinar as descrições em um único DataFrame
dados = pd.DataFrame(descricoes_preguica + descricoes_outros)
X = dados['descrição']
y = dados['rótulo']

# Passo 2: Pré-processamento e divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 3: Representação dos textos
vetorizar = TfidfVectorizer()
X_train_tfidf = vetorizar.fit_transform(X_train)
X_test_tfidf = vetorizar.transform(X_test)

# Passo 4: Construção e treinamento do modelo NLP
modelo_nlp = MultinomialNB()
modelo_nlp.fit(X_train_tfidf, y_train)

# Passo 5: Avaliação do modelo NLP
y_pred = modelo_nlp.predict(X_test_tfidf)
print(f'Acurácia: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Passo 6: Implementação da inferência
def prever_animal(texto):
    texto_tfidf = vetorizar.transform([texto])
    return modelo_nlp.predict(texto_tfidf)[0]

# Dicionário de mapeamento de rótulos para nomes em português
traducao_classes = {
    'cat': 'gato',
    'dog': 'cachorro',
    'sloth': 'bicho-preguiça'
}

# Carrega o modelo de CNN
model = load_model('sloth-recognizer/Image_classify.keras')
data_cat = ['cat', 'dog', 'sloth']
img_height = 180
img_width = 180
st.header('Modelo de Visão Computacional')
image_path = st.text_input('Insira o nome da imagem:', './sloth-recognizer/Sloth.jpg')

# Carrega e processa a imagem
image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_arr = tf.keras.preprocessing.image.img_to_array(image_load)
img_bat = tf.expand_dims(img_arr, 0)

# Faz a predição da imagem
predict = model.predict(img_bat)

# Calcula a pontuação
score = tf.nn.softmax(predict[0])

# Mostra a imagem e a predição
st.image(image_path)
classe_predita = data_cat[np.argmax(score)]
classe_predita_pt = traducao_classes[classe_predita]
st.write(f'O animal na imagem é um {classe_predita_pt}')
st.write(f'Com a precisão de {100 * np.max(score):.2f}%')

# Entrada de texto para o modelo NLP
st.header('Modelo de Processamento de Linguagem Natural')
texto_input = st.text_area('Insira uma descrição do animal para a classificação de NLP:', 'Este animal se move lentamente e passa a maior parte do tempo nas árvores.')
if st.button('Classificar Descrição'):
    resultado = prever_animal(texto_input)
    st.write(f'O texto descreve: {resultado}')
