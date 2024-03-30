# Questão 1 - Letra a - Solução Analítica 

from sympy import symbols, Function, Eq, Derivative, sqrt, dsolve, simplify
import numpy as np 
import math as mt 
import matplotlib.pyplot as plt 

x = symbols('x')
y = Function('y')(x)

eq = Eq(Derivative(y,x), (1+4*x)*sqrt(y))

solucao_analitica = simplify(dsolve(eq,y))

print(solucao_analitica)


