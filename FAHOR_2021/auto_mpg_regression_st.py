# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 10:39:23 2021

@author: filip

Criar uma RNA para prever a qualidade do vinho baseado em suas características

"""
#Importando bibliotecas

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler # normaliza entradas e saídas
from sklearn.model_selection import train_test_split # divide os dados p/ treinamento e teste
from sklearn.metrics import r2_score # métrica para medir o treinamento
from sklearn.neural_network import MLPRegressor # ANN em si - RNA em si


#Importação do dataset
dataset = pd.read_excel("Auto MPG.xlsx")

#Conversão do dataset para formato numpy
data = np.array(dataset)

#Definição das entradas e das saídas
Entradas = data[:,2:]
Saida = data[:,0]
Saida = np.reshape(Saida,(len(Saida),1))

#Variáveis auxiliares
I = np.size(Entradas,1)
O = np.size(Saida,1)

escalai = StandardScaler()
escalao = StandardScaler()

escalai.fit(Entradas) # cria padrão de normalizaçaõ
Entradas_n = escalai.transform(Entradas) # normaliza os dados
escalao.fit(Saida) # cria padrão de normalizaçaõ
Saida_n = escalao.transform(Saida) # normaliza os dados

#Criando conjunto de dados para treinamento (train) e validação (test)
Entradas_n_treinamento, Entradas_n_teste, Saida_n_treinamento, Saida_n_teste = train_test_split(Entradas_n,Saida_n, test_size = 0.1)
S = len(Entradas_n_teste)

#Criando a RNA

RNA = MLPRegressor(hidden_layer_sizes=(100), # Número de camadas ocultas e 
                    max_iter=1000, # Número máximo de interações
                    tol=0.00001, # Tolerância para interromper o treinamento
                    learning_rate_init=0.001, # alpha(0)
                    solver='adam', # Algoritmo de treinamento - otimização
                    activation='relu', #Função de ativação
                    learning_rate='adaptive', # Utilizar alpha adaptativo
                    n_iter_no_change=1000, # parâmetro para encerrar o treinamento
                    early_stopping = True, # Parada precoce, a fim de evitar overfitting
                    validation_fraction = .2, # Divisão dos dados de treinamento
                    verbose=0)

RNA.fit(Entradas_n_treinamento,Saida_n_treinamento)

Predicao_n = np.reshape(RNA.predict(Entradas_n_teste),(S,1))
Predicao = escalao.inverse_transform(Predicao_n)
Real_valores_n = np.reshape(Saida_n_teste,(S,1))
Real_Valores = escalao.inverse_transform(Real_valores_n)

R2RNA = r2_score(Real_Valores,Predicao)
MAE = np.mean(np.abs(Real_Valores-Predicao))
print("\n")
print("Erro médio absoluto", MAE)
print("R2: ", R2RNA)


st.header("Teste de Front-end")
st.write("Primeiro Front-end para Machine Learning")
st.sidebar.write("Aqui você pode escrever na barra lateral")


Cilindros = st.selectbox("Selecione quantos cilindros",[3,4,5,6,8],0)
Distancia = st.number_input("Digite a Distância entre 68 e 455",68,455,100)
horsepower = st.number_input("Digite a potência - HP - entre 46 e 230",46,230,100)
weight = st.number_input("Digite o peso em lb - entre 1613 e 5140",1613,5140,2000)
acceleration = st.number_input("Digite a aceleração - entre 8 e 25",8,25,10)
modelyear = st.slider("Selecione o modelo do ano",70,82,76)
Origem = st.radio("Escolha o país de origem",[1,2,3])

# clicked = st.button("Rodar predição")
# if clicked == 1: 
    
# print(np.reshape([Cilindros,Distancia,horsepower,weight,acceleration,modelyear,Origem],(1,I)))
    
Entrada_st = escalai.transform(np.reshape([Cilindros,Distancia,horsepower,weight,acceleration,modelyear,Origem],(1,I)))
Predicao_st = RNA.predict(Entrada_st)
Predicao = escalao.inverse_transform(np.reshape(Predicao_st,(1,O)))

st.write("O valor de mpg previsto é de: ", Predicao[0,0])