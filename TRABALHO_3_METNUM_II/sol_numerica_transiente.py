import numpy as np
import matplotlib.pyplot as plt
import time

def Gauss_Seidel(A, b, x0, Eppara, maxit):
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

# Critério de Scarvorought, 1966
n = 12 # Números de algarismos significativos
Eppara = 0.5*10**(2-n) # Termo relativos

# Número Máximo de Iterações
maxit = 1000

# Propriedades do Material - Cobre
rho = 8.92  # g/cm^3
cp = 0.092  # cal/(g.ºC)
k = 0.95  # cal/(cm.s.ºC)
Lx = 100  # cm
Ly = 100  # cm

# Dados Iniciais
tempo_maximo = 1000  # segundos
To = 0  # ºC
Tn = 20  # ºC
Ts = 0  # ºC
Tw = 0  # ºC
Te = 0  # ºC

# Parâmetros de simulação
nx = 10
ny = 10
N = nx * ny
nt = 1000

# Cálculos Iniciais
alpha = k / (rho * cp)
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = tempo_maximo / nt
rxt = alpha * dt / dx**2
ryt = alpha * dt / dy**2

# Solução da Equação do Calor - MDFI
inicio = time.time()

# Condição Inicial
T_old = np.full((nx, ny), To)
T_new = np.full((nx, ny), To)
Told = np.full(N,To)

# Inicializando as Matrizes
A = np.zeros((N, N))
B = np.zeros(N)
tempo = 0
h = 0

# Definindo os índices da matriz 2D para o vetor 1D
ind = np.arange(N).reshape(nx,ny)

while tempo < tempo_maximo:
    h += 1

    # Parte central
    for i in range(1,ny-1):
      for m in range(i*nx+1,(i+1)*nx-1):
        Ap = 1 + 2*rxt + 2*ryt
        Aw = -rxt
        Ae = -rxt
        As = -ryt
        An = -ryt
        S = Told[m]

        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m+1] = Ae
        A[m,m+nx] = As
        A[m,m-nx] = An
        B[m] = S

    # Canto Superior Esquerdo
    m = 0
    Ap = 1 + 4*rxt + 4*ryt
    Ae = -4/3*rxt
    As = -4/3*ryt
    S = Told[m] + 8/3*Tw*rxt + 8/3*Tn*ryt

    A[m,m] = Ap
    A[m,m+1] = Ae
    A[m,m+nx] = As
    B[m] = S

    # Fronteira Norte
    for m in range(1,nx-1):
      Ap = 1 + 2*rxt + 4*ryt
      Aw = -rxt
      Ae = -rxt
      As = -4/3*ryt
      S = Told[m] + 8/3*Tn*ryt

      A[m,m] = Ap
      A[m,m-1] = Aw
      A[m,m+1] = Ae
      A[m,m+nx] = As
      B[m] = S

    # Canto Superior Direito
    m = nx-1
    Ap = 1 + 4*rxt + 4*ryt
    Aw = -4/3*rxt
    As = -4/3*ryt
    S = Told[m] + 8/3*Te*rxt + 8/3*Tn*ryt

    A[m,m] = Ap
    A[m,m-1] = Aw
    A[m,m+nx] = As
    B[m] = S

    # Fronteira Oeste
    for m in range(nx,(ny-2)*nx+1,nx):
      Ap = 1 + 2*rxt + 4*ryt
      Ae = -4/3*rxt
      An = -ryt
      As = -ryt
      S = Told[m] + 8/3*Tw*rxt

      A[m,m] = Ap
      A[m,m+1] = Ae
      A[m,m+nx] = As
      A[m,m-nx] = An
      B[m] = S

    # Fronteira Leste
    for m in range(2*nx-1,(ny-1)*nx,nx):
      Ap = 1 + 2*rxt + 4*ryt
      Aw = -4/3*rxt
      An = -ryt
      As = -ryt
      S = Told[m] + 8/3*Te*rxt

      A[m,m] = Ap
      A[m,m-1] = Aw
      A[m,m-nx] = An
      A[m,m+nx] = As
      B[m] = S

    # Canto Inferior Esquerdo
    m = (ny-1)*nx
    Ap = 1 + 4*rxt + 4*ryt
    Ae = -4/3*rxt
    As = -4/3*ryt
    S = Told[m] + 8/3*Tw*rxt + 8/3*Ts*ryt

    A[m,m] = Ap
    A[m,m+1] = Ae
    A[m,m-nx] = An
    B[m] = S

    # Fronteira Sul
    for m in range((ny-1)*nx+1,ny*nx-1):
      Ap = 1 + 2*rxt + 4*ryt
      Aw = -rxt
      Ae = -rxt
      An = -4/3*ryt
      S = Told[m] + 8/3*Ts*ryt

      A[m,m] = Ap
      A[m,m-1] = Aw
      A[m,m+1] = Ae
      A[m,m-nx] = An
      B[m] = S

    # Canto Inferior Direito
    m = ny*nx-1
    Ap = 1 + 4*rxt + 4*ryt
    Ae = -4/3*rxt
    As = -4/3*ryt
    S = Told[m] + 8/3*Te*rxt + 8/3*Ts*ryt

    A[m,m] = Ap
    A[m,m-1] = Aw
    A[m,m-nx] = An
    B[m] = S

    # Solução do sistema Linear
    x0 = np.ones(N)
    T = Gauss_Seidel(A, B, x0, Eppara, maxit)

    T_new = np.zeros((nx, ny), dtype=np.float64)
    # Extração do valor da variável T
    for i in range(nx):
        for j in range(ny):
            T_new[i, j] = T[ind[i, j]]

    Told = T.copy()
    T_old = T_new.copy()
    tempo += dt


# Plot
plt.figure()
plt.imshow(T_new, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.title('Distribuição de Temperatura - Solução Transiente')

plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
ax.plot_surface(X, Y, T_new.T, cmap='viridis')
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Temperatura (ºC)')
plt.title('Superfície de Temperatura - Solução Transiente')

plt.show()

print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
