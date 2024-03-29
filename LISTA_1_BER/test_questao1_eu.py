import numpy as np
import matplotlib.pyplot as plt

def fxy(x, y):
    return (1 + 4 * x) * np.sqrt(y)

def analytical_solution(x):
    return ((x + 2 * x**2 + 2) / 2)**2 #(x**2 + 4*x**3 + 4*x**4)/4

#dados iniciais
a = 0
b = 1
h = 0.25
n = int((b - a) / h) + 1

#listas
x = [a]
y = [1]
error = []

#EULER

for i in range(n - 1):
    x_next = x[i] + h
    y_next = y[i] + h * fxy(x[i], y[i])
    x.append(x_next)
    y.append(y_next)
    y_analytical = analytical_solution(x_next)
    error_next = abs((y_analytical - y_next) / y_next) * 100
    error.append(error_next)

plt.figure()
plt.plot(x, y, label='Numérico')
plt.plot(x, [analytical_solution(xi) for xi in x], label='Solução Analítica', color='black', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Euler vs Solução Analítica')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x[:-1], error, label='Erro do Método de Euler (%)', color='red', linestyle='dashdot')
plt.xlabel('x')
plt.ylabel('Erro (%)')
plt.title('Erro do Método de Euler')
plt.legend()
plt.grid()
plt.show()

error_rounded = [round(value, 4) for value in error]
print("Erros para cada iteração:", error_rounded)

# RUNGE-KUTTA DE 2º ORDEM
a = 0
n = 5
h = 0.25

x = [a]
y = [1]
error = []

for i in range(n - 1):
    x_next = x[i] + h
    k1 = h * fxy(x[i], y[i])
    k2 = h * fxy(x[i] + h / 2, y[i] + k1 / 2)
    y_next = y[i] + k2
    x.append(x_next)
    y.append(y_next)
    y_analytical = analytical_solution(x_next)
    error_next = abs((y_analytical - y_next) / y_next) * 100
    error.append(error_next)

plt.figure()
plt.plot(x, y, label='RK2 Numérico')
plt.plot(x, [analytical_solution(xi) for xi in x], label='Solução Analítica', color='black', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title('RK2 Numérico vs Solução Analítica')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x[:-1], error, label='Erro do Método RK2 (%)', color='red', linestyle='dashdot')
plt.xlabel('x')
plt.ylabel('Erro (%)')
plt.title('Erro do Método RK2')
plt.legend()
plt.grid(True)
plt.show()

error_rounded = [round(value, 4) for value in error]
print("Erros para cada iteração:", error_rounded)

# RUNGE-KUTTA DE 4º ORDEM

a = 0
n = 5
h = 0.25

x = [a]
y = [1]
error = []

for i in range(n - 1):
    x_next = x[i] + h
    k1 = h * fxy(x[i], y[i])
    k2 = h * fxy(x[i] + h / 2, y[i] + k1 / 2)
    k3 = h * fxy(x[i] + h / 2, y[i] + k2 / 2)
    k4 = h * fxy(x[i] + h, y[i] + k3)
    y_next = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    x.append(x_next)
    y.append(y_next)
    y_analytical = analytical_solution(x_next)
    error_next = abs((y_analytical - y_next) / y_next) * 100
    error.append(error_next)

plt.figure()
plt.plot(x, y, label='RK4 Numérico')
plt.plot(x, [analytical_solution(xi) for xi in x], label='Solução Analítica', color='black', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.title('RK4 Numérico vs Solução Analítica')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x[:-1], error, label='Erro do Método RK4 (%)', color='red', linestyle='dashdot')
plt.xlabel('x')
plt.ylabel('Erro (%)')
plt.title('Erro do Método RK4')
plt.legend()
plt.grid(True)
plt.show()

error_rounded = [round(value, 4) for value in error]
print("Erros para cada iteração:", error_rounded)

'''
#TODOS MÉTODOS JUNTOS

plt.figure()
plt.plot(x, y, label='Numérico')
plt.plot(x, [analytical_solution(xi) for xi in x], label='Solução Analítica', color='black', linestyle='dashed')
plt.plot(x, y, label='RK2 Numérico')
plt.plot(x, y, label='RK4 Numérico')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Euler vs Solução Analítica')
plt.legend()
plt.grid()
plt.show()

#TODOS ERROS JUNTOS

plt.figure()
plt.plot(x[:-1], error, label='Erro do Método de Euler (%)', color='red', linestyle='dashdot')
plt.xlabel('x')
plt.ylabel('Erro (%)')
plt.title('Erro do Método de Euler')
plt.legend()
plt.grid()
plt.show()
'''