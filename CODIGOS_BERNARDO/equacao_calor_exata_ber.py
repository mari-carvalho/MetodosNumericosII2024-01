# Implementação da Equação do Calor
# Bernardo Skovronski Woitas
# Data: 04/04/2024

# Bibliotecas
import numpy as np
import math
import matplotlib.pyplot as plt

# Propriedades do Material - Cobre
rho = 8.92 # g/cm^3
cp = 0.092 # cal/(g.ºC)
k = 0.95 # cal/(cm.s.ºC)
L = 80 # cm
N = 200 # MNúmeros de termos da Série de Fourier

# Cálculos Iniciais
c_2 = k/(rho * cp) # cm^2/s (difusividade térmica)
lambda_n = np.sqrt(c_2) * np.pi/L

# Define a função desejada:
def f(x): #função inicial
    return 20

# Coeficiente da Série de Fourier:
def fourier(f, L, N):
    Bn = np.zeros(N) #pré alocando para colocar cada valor de nzinho para o Bn

    for n in range (1, N):
        integrando = lambda x: f(x) * np.sin(n * np.pi *x /L) # define a função em linha, (integrando = parte interna da integral)

        Bn[n]= 2/L * np.trapz(integrando(np.linspace(0, L, 1000)), np.linspace(0, L, 1000)) # cria um vetor (linspace) para alocar os valores de 0 a L com 1000 divisões. O np.trapz precisa do vetor que estamos usando, por isso repete o vetor do integrando

    return Bn

# solução da Equação do Calor:
def calor(x, t, L, N):
    solucao = np.zeros_like(x) #para cada valor de x, teremos uma solução
    coef = fourier(f, L, N) #retorna um vetor

    for n in range (1, N):
        solucao += coef[n] * np.sin(n * np.pi * x/L) * np.exp(-lambda_n**2 * n * t)

    return solucao

# Define Valores de x e t:
x = np.linspace(0, L, 100)
t = np.linspace(0, 1000, 100)

X, T = np.meshgrid(x,t)

Temperatura = calor(X, T, L, N)

print(Temperatura)

# Gráficos:
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, Temperatura, rstride=2, cstride=2, cmap=plt.cm.viridis, linewidth=0.5, antialiased=True)
ax.set_xlabel=('x (cm)')
ax.set_ylabel=('t (segundos)')
#ax.set_zlabel=('T (x,y) (ºC)')
plt.title('Equação do Calor')
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(30,60)
plt.show()