# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 09:45:49 2021

@author: Filipe Pacheco

Criar uma RNA capaz de classificar animais em um Zoo baseado em suas características

"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_excel("zoo.xlsx")

data = np.array(dataset)

# Mammal -> 0
# Fish -> 1
# Bird -> 2
# Invertebrate -> 3
# Amphibian -> 4
# Reptile -> 5
# Insect -> 6 

classe = ["mammal","fish","bird","invertebrate","amphibian","reptile","insect"]

#Transformar as saídas em valores numéricos

for i in range(len(data)):
    for j in range(len(classe)):
        if data[i,0] == classe[j]:
            data[i,0] = j
            
#Transformar as demais características em valores numéricos
for i in range(len(data)):
    for j in range(2,18):
        if data[i,j] == "yes":
            data[i,j] = 1
        if data[i,j] == "no":
            data[i,j] = 0

# Separar as entradas do dataset
Entradas = data[:,2:18]

# Separar as saídas do dataset
Saida = data[:,0]

# Dividir os dados em conjunto de treinamento(train) e validação(test)
Entradas_treinamento, Entradas_teste, Saida_treinamento, Saida_teste = train_test_split(Entradas, Saida, test_size=.1)

Saida_treinamento= np.array(np.reshape(Saida_treinamento,(90,)), dtype=np.int32)

# Criando a RNA e definindo seus parâmetros
Clf = MLPClassifier(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(15,))

#Treinando a RNA
Clf.fit(Entradas_treinamento,Saida_treinamento)

Predicao = Clf.predict(Entradas_teste)
print("\n")
print("Predição da RNA",Clf.predict(Entradas_teste))

print("Dados reais    ",Saida_teste)

