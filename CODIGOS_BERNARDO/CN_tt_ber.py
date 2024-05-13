# Bibliotecas:
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Propreidades do Material - Cobre:
rho = 8.92 # g/cm³
cp = 0.092 # cal/(g.ºC)
k = 0.95 # cal/(cm.s.ºC)
L = 80 # cm
T0 = 20 # ºC
Tw = 0 # ºC
Te = 0 # ºC
h_x = 0.25
h_t = 0.5
t0 = 0
tf = 100
x0 = 0 
xf = L

# Dados Iniciais:
tempo = 1000 #s


# Malha:
nx = 50
nt = 2000
t = np.zeros(int(nt)+1) 
# Calculos Iniciais:
alpha = k/(rho*cp) #difusividade térmica (m²/s)
dx = L/nx
dt=tempo/nt
rxt = alpha*(dt/(dx**2))

# Critérios de Scarvorought, 1966:
n = 12 
Eppara = 0.5*10**-n

# Número Máximo de Interações:
maxit = 1000
ai = -1/2*rxt 
bi = 1 + rxt
an = -(2/3)*rxt
b1 = 1 + 2*rxt

def Gauss_Seidel(A, b, x0, Eppara, maxit): #matriz a, vetor b, chute inicial x0, critério de parada Eppara e Número Máximo de Iterações maxit
    ne = len(b)
    x = np.zeros(ne) if x0 is None else np.array(x0)


    iter = 0

    Epest = np.linspace(100,100,ne)

    while np.max(Epest) >= Eppara and iter <= maxit:
        x_old = np.copy(x)

        for i in range(ne):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Critério de parada
        Epest = np.abs((x - x_old) / x) * 100

        iter += 1

    return x

#Inicialização das Matrizes
T_matriz = np.zeros((nx, nx))
T_old = np.ones(nx) * T0 # T_old vai sendo atualizado, por isso é um vetor de 1, porque se multiplicasse Temperatura por zeros não daria certo
T = np.zeros ((nt+1, nx)) # Matriz Solução
D = np.zeros(nx) # Resultado do produto matricial A * T 
h = 0 # Contador para iteração
T[h, :] = T0 # Contador para todas as colunas
t = 0 # Tempo

#
for j in range(nx): # 0 a 4 (tamanho de t), tempo 4 elemento 5; 1 a 4 (tamanho de t, mesmo for), tempo 4 elemento 4; precisa de mais um elemento
            h = h + 1 
            T_matriz = np.zeros((nx, nx))
            for i in range(len(T_matriz)): # variando a linha
                if i == 0:
                    T_matriz[i,0] = b1
                    T_matriz[i,1] = an
                    D[i] = (1 - 2*rxt)*T_old[i] + (2/3*rxt)*T_old[i+1] + 8/3*rxt*Te
                elif i == len(T_matriz)-1: # o último, N
                    T_matriz[i,len(T_matriz)-2] = an
                    T_matriz[i,len(T_matriz)-1] = b1
                    D[i] = (1 - 2*rxt)*T_old[i] + (2/3*rxt)*T_old[i-1] + 8/3*rxt*Tw
                else:
                    T_matriz[i,i-1] = ai # linha 1, coluna 0 (i-1)
                    T_matriz[i,i] = bi
                    T_matriz[i,i+1] = ai
                    D[i] = (1 - rxt)*T_old[i] + (1/2*rxt)*T_old[i+1] + (1/2*rxt)*T_old[i-1]# condição central é 0

x0 = T_old
T_new = Gauss_Seidel(T_matriz, D, x0, Eppara, maxit)
T_old = T_new # Para substituir o old para prosseguir calculando
t = t + dt
T[h, :] = T_new

    # Plotagem

X = np.linspace(0, L, nx)
Y = np.linspace(0, tempo, nt+1)
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, T, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
ax.set_xlabel('x (cm)')
ax.set_ylabel('t (segundos)')
ax.set_zlabel('T(x,t) (ºC)')
plt.title('Equação do Calor')
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(30, 30)
plt.show()

