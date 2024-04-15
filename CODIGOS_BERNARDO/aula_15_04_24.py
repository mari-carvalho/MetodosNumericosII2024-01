import numpy as np
import math
import matplotlib.pyplot as plt

#Propreidades do Material - Cobre:
rho = 8.92 # g/cm³
cp = 0.092 # cal/(g.ºC)
k = 0.95 # cal/(cm.s.ºC)
L = 80 # cm

# Dados Iniciais:
t = 1000 #s
T0 = 20 # ºC
Tw = 0 # ºC
Te = 0 # ºC

# Malha:
nx = 50
nt = 2000

# Calculos Iniciais:
alpha = k/(rho*cp) #difusividade térmica (m²/s)
dx = L/nx
dt=t/nt
rxt = alpha*(dt/(dx**2))

# Análise de stabilidade e Conergência:
if dt <= (dx**2) / (4 * alpha):
    print("Método Convergente")
else:
    print("Método Não Convergente. Refina a malha do tempo 'nt' ")
    
# Solução da Equação do Calor 1D Transiente - MDFE:
# Condição inicial:
T = np.zeros((nt+1, nx))

for i in range (nx):
    T[0,i]=T0

# Equações Discretizadas
for n in range (0,nt): # varia com o tempo (primeiro a vir, representando linha)
    for i in range (0,nx): # varia com  espaço (segundo a vir, representando coluna)
        if i == 1:
            T[n+1, i] = 8/3 * rxt * Tw + (1 - 4 * rxt) * T[n, i] + 4/3 * rxt * T[n, i + 1]
        elif i == nx-2:
            T[n+1, i] = 4/3 * rxt * T[n, i - 1] + (1 - 4 * rxt) * T[n, i] + 8/3 * rxt * Te
        elif i >= 1 and i < nx-2:
            T[n+1, i] = rxt * T[n, i-1] + (1 - 2 * rxt) * T[n, i] + rxt * T[n, i+1]

print(T)

# Gráficos:
# Define Valores de x e t:
x = np.linspace(0, L, nx)
y = np.linspace(0, t, nt+1)

X, Y = np.meshgrid(x,y)



# Gráficos:
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, T, rstride=2, cstride=2, cmap=plt.cm.viridis, linewidth=0.5, antialiased=True)
ax.set_xlabel=('x (cm)')
ax.set_ylabel=('t (segundos)')
#ax.set_zlabel=('T (x,y) (ºC)')
plt.title('Equação do Calor')
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(30,80)
plt.show()