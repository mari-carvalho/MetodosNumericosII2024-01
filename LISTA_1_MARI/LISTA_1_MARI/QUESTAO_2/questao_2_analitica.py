# Questão 2 - Solução Analítica

from sympy import symbols, Eq, Function, Derivative, dsolve, simplify
import numpy as np 

x = symbols('x')
y = Function('y')(x)

eq = Eq(Derivative(y,x), -2*y + x**2)

solucao_analitica = simplify(dsolve(eq,y))

print(solucao_analitica)