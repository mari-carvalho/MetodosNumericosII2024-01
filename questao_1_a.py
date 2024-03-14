# Resolução Analítica:

import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

#Definindo Intervalo de Tempo:
t = [0,1,2]

# Solução Exata:
def calculate_yexata(t:float) -> float:

    y_exata = np.exp(((t[i]**4)/4) - 1.5*t[i])

    return y_exata 

list_yexata = []

for i in range(len(t)):

    y_exata = calculate_yexata(t)
    list_yexata.append(y_exata)
    
print(list_yexata)

#Plotando a Solução Exata:
plt.plot(t, list_yexata, marker='o', linestyle='-', color='#7B2791', label='Solução Analítica/Exata')

plt.title('Resolução de PVI')
plt.xlabel('tempo(s)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()

