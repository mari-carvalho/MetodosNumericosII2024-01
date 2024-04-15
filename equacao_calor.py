import numpy as np 
import sympy as sp 
import math as mt 
import matplotlib.pyplot as plt 

# Propriedades do Material - Cobre:
rho = 8.92 # g/cm³
cp = 0.092 # cal/(g*°C)
k = 0.95 # cal/(cm*s*°C)
L = 80 #cm 
N = 200 # número de termos da série de Fourier

# Cálculos Iniciais:
def calculate_c2(k:float, rho:float, cp:float):

    c2 = k/(rho*cp)

    return c2 

c2 = calculate_c2(k,rho,cp)

def calculate_lambda_n(c2: float, L:float):
        
    lambda_n = (np.sqrt(c2)*np.pi)/L

    return lambda_n

lambda_n = calculate_lambda_n(c2,L)
print(lambda_n)

# Definição da Função Desejada:
def f(x):
    return 20

# Coeficiente da Série de Fourier:
def calculate_Bn(f, L:float, N:float):

    Bn = np.zeros(N)
    for i in range(1,N):
        integrando = lambda x: f(x)*np.sin((i*np.pi*x)/L)
        Bn[i] = (2/L) * np.trapz(integrando(np.linspace(0,L,1000)), np.linspace(0,L,1000))

    return Bn 
    
# Solução da Equação do Calor:
def calculate_calor(x:np.ndarray, t, N:float, L:float):

    calor = np.zeros_like(x)
    Bn = calculate_Bn(f,L,N)

    for i in range(1,N):
        calor += Bn[i]*np.sin((i*np.pi*x)/L) * np.exp(-(i*lambda_n**2)*t)

    return calor

# Define valores de x e t:

x = np.linspace(0,L,100)
t = np.linspace(0,1000,100)

X, T = np.meshgrid(x,t)

Temperatura = calculate_calor(X,T,N,L)
print(Temperatura)

# Plotagem dos Dados:

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, Temperatura, rstride=2, cstride=2, cmap=plt.cm.viridis, linewidth=0.5, antialiased=True)
ax.set_xlabel('x (cm)')
ax.set_ylabel('t (s')
ax.set_zlabel('T(x,y) (°C)')
plt.title('Equação do Calor')
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(30,60)

plt.show()