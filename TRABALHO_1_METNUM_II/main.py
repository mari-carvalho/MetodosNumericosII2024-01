# Main do Trabalho 1 de Métodos Numéricos II:

# Importando as Bibliotecas 
import numpy as np 
import scipy as sp
import math as mt 
import matplotlib.pyplot as plt 
from QUESTAO_1 import questao_1_a, questao_1_b, questao_1_c_rk2, questao_1_c_rk4
from QUESTAO_3 import questao_3_euler, questao_3_rk2, questao_3_rk4

# Questão 1:
# Letra A - Analiticamente:
letra_a = calculate_letra_a()
print(letra_a)

# Letra B - Método de Euler para h = 0.5 e 0.25:
letra_b = calculate_letra_b()
print(letra_b)

# Letra C - Método de Runge Kutta de 2ª e 4ª Ordem para h = 0.5:
letra_c_rk2 = calculate_letra_c_rk2()
print(letra_c_rk2)

letra_c_rk4 = calculate_letra_c_rk4()
print(letra_c_rk4)

# Questão 3:
# Sistema de EDO com Método de Euler:
euler = calculate_3_euler()
print(euler)

# Sistema de EDO com Método de Runge-Kutta de 2ª Ordem:
rk2 = calculate_3_rk2()
print(rk2)

# Sistema de EDO com Método de Runge-Kutta de 4ª Ordem:
rk4 = calculate_3_rk4()
print(rk4)
