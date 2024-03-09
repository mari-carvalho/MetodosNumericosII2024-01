# Resolução do Problema de PVI:
#Importando Bibliotecas:
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

# Solução Numérica:

t0 = 0
tf = 4
y0 = 2
n = 5 #número de elementos 
h = (tf-t0)/(n-1) # passo

def f1(t,y): # Define a Função EDO
    
    return 4*np.exp(0.8*t) - 0.5*y

t = np.zeros(n)
y = np.zeros(n)

for i in range(0,n):
    if i == 0:
        y[i] = 2
        t[i] = t0
    else:
        t[i] = i*h
        y[i] = 0 # só para alimentar o vetor com zeros e depois varrer e ocupar os espaços com novos valores 

# Método de Euler:

for i in range(0,n-1): # precisa ser n-1 para ir de 0 a 4 (5 elementos)
    #print(y[i]) # o valor anterior y[i] vai ser usado para somar com a função e gerar o novo valor da próxima interação y[i+1]
    y[i+1] = y[i] + h*f1(t[i],y[i])
print(y[i+1])
yex = (4/1.3) * (np.exp(0.8*t) - np.exp(-0.5*t)) + 2*np.exp(-0.5*t)

plt.plot(t, y, marker='o', linestyle='-', color='#7B2791', label='Solução Numérica')
plt.plot(t, yex, marker='o', linestyle='-', color='blue', label='Solução Analítica/Exata')
plt.title('Exemplo 1')
plt.xlabel('tempo(s)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

# Calculando o Erro Percentual Verdadeiro - Comparação de Analítica e Numérica:

erro = np.abs((yex-y)/yex)*100
print(erro)

plt.plot(t, erro, marker='o', linestyle='-', color='red', label='Erro Percentual Verdadeiro [%]')
plt.title('Erro Percentual Verdadeiro [%]')
plt.xlabel('tempo(s)')
plt.ylabel('Erro [%]')
plt.grid(True)
plt.legend()
plt.show()

