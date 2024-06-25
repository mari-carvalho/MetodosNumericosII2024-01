import numpy as np
import matplotlib.pyplot as plt
import time
from solvers import solvers 

class temp_comp():
    
    def calculate_gs(nx):
    
        # Critério de Scarvorought, 1966
        n = 12  # Números de algarismos significativos
        Eppara = 0.5 * 10 ** (2 - n)  # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 5  # cm
        Ly = 5  # cm
        L = Lx
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 0  # ºC
        Tn = 0 # ºC
        Ts = 20  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        q = 20000
        
        # Parâmetros de simulação
        nx = nx
        ny = nx
        N = nx * ny
        nt = 10
        
        # Cálculos Iniciais
        alpha = k / (rho * cp)
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)
        dt = tempo_maximo / nt
        rxt = alpha * dt / dx ** 2
        ryt = alpha * dt / dy ** 2
        
        # Solução da Equação do Calor - MDFI
        inicio = time.time()
        
        # Condição Inicial
        T_old = np.full((nx, ny), To)
        T_new = np.full((nx, ny), To)
        Told = np.full(N, To)
        
        # Inicializando as Matrizes
        A = np.zeros((N, N))
        B = np.zeros(N)
        tempo = 0
        h = 0
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx, ny)
        
        while tempo < tempo_maximo:
            h += 1
        
            # Parte central
            for i in range(1, ny - 1):
                for m in range(i * nx + 1, (i + 1) * nx - 1):
                    Ap = 1 + 2 * rxt + 2 * ryt
                    Aw = -rxt
                    Ae = -rxt
                    As = -ryt
                    An = -ryt
                    S = Told[m]
        
                    A[m, m] = Ap
                    A[m, m - 1] = Aw
                    A[m, m + 1] = Ae
                    A[m, m + nx] = As
                    A[m, m - nx] = An
                    B[m] = S
        
            nxl = nx / L
            p = nxl * (nx + ny) + nxl * 2
            l = p + nxl - 1
            for b in range(0, int(nxl)):
                for j in range(int(p) + (b * nx), int(l) + (b * nx) + 1):
                    B[j] = Told[j] + q
        
            # Canto Superior Esquerdo
            m = 0
            Ap = 1 + 4 * rxt + 4 * ryt
            Ae = -4 / 3 * rxt
            As = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Tn * ryt
        
            A[m, m] = Ap
            A[m, m + 1] = Ae
            A[m, m + nx] = As
            B[m] = S
        
            # Fronteira Norte
            for m in range(1, nx - 1):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -rxt
                Ae = -rxt
                As = -4 / 3 * ryt
                S = Told[m] + 8 / 3 * Tn * ryt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m + 1] = Ae
                A[m, m + nx] = As
                B[m] = S
        
            # Canto Superior Direito
            m = nx - 1
            Ap = 1 + 4 * rxt + 4 * ryt
            Aw = -4 / 3 * rxt
            As = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Tn * ryt
        
            A[m, m] = Ap
            A[m, m - 1] = Aw
            A[m, m + nx] = As
            B[m] = S
        
            # Fronteira Oeste
            for m in range(nx, (ny - 2) * nx + 1, nx):
                Ap = 1 + 2 * rxt + 4 * ryt
                Ae = -4 / 3 * rxt
                An = -ryt
                As = -ryt
                S = Told[m] + 8 / 3 * Tw * rxt
        
                A[m, m] = Ap
                A[m, m + 1] = Ae
                A[m, m + nx] = As
                A[m, m - nx] = An
                B[m] = S
        
            # Fronteira Leste
            for m in range(2 * nx - 1, (ny - 1) * nx, nx):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -4 / 3 * rxt
                An = -ryt
                As = -ryt
                S = Told[m] + 8 / 3 * Te * rxt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m - nx] = An
                A[m, m + nx] = As
                B[m] = S
        
            # Canto Inferior Esquerdo
            m = (ny - 1) * nx
            Ap = 1 + 4 * rxt + 4 * ryt
            Ae = -4 / 3 * rxt
            An = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Ts * ryt
        
            A[m, m] = Ap
            A[m, m + 1] = Ae
            A[m, m - nx] = An
            B[m] = S
        
            # Fronteira Sul
            for m in range((ny - 1) * nx + 1, ny * nx - 1):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -rxt
                Ae = -rxt
                An = -4 / 3 * ryt
                S = Told[m] + 8 / 3 * Ts * ryt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m + 1] = Ae
                A[m, m - nx] = An
                B[m] = S
        
            # Canto Inferior Direito
            m = ny * nx - 1
            Ap = 1 + 4 * rxt + 4 * ryt
            Aw = -4 / 3 * rxt
            An = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Ts * ryt
        
            A[m, m] = Ap
            A[m, m - 1] = Aw
            A[m, m - nx] = An
            B[m] = S
        
            # Solução do sistema Linear
            x0 = np.ones(N)
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
            T_new = np.zeros((nx, ny), dtype=np.float64)
            # Extração do valor da variável T
            for i in range(nx):
                for j in range(ny):
                    T_new[i, j] = T[ind[i, j]]
            
            T_new_inv = T_new[::-1]
        
            Told = T.copy()
            T_old = T_new.copy()
            tempo += dt
        
        # Plot
        plt.figure()
        plt.imshow(T_new_inv, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'Distribuição de Temperatura - Termo Fonte - Gauss Seidel - nx={nx}')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new_inv.T, cmap='viridis')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Temperatura [°C]')
        plt.title(f'Superfície de Temperatura - Termo Fonte - Guass Seidel - nx={nx}')
        
        plt.show()
        
        temp_simu = time.time() - inicio
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return T_new_inv, temp_simu
        
    def calculate_jac(nx):
    
        # Critério de Scarvorought, 1966
        n = 12  # Números de algarismos significativos
        Eppara = 0.5 * 10 ** (2 - n)  # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 5  # cm
        Ly = 5  # cm
        L = Lx
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 0  # ºC
        Tn = 0 # ºC
        Ts = 20  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        q = 20000
        
        # Parâmetros de simulação
        nx = nx
        ny = nx
        N = nx * ny
        nt = 10
        
        # Cálculos Iniciais
        alpha = k / (rho * cp)
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)
        dt = tempo_maximo / nt
        rxt = alpha * dt / dx ** 2
        ryt = alpha * dt / dy ** 2
        
        # Solução da Equação do Calor - MDFI
        inicio = time.time()
        
        # Condição Inicial
        T_old = np.full((nx, ny), To)
        T_new = np.full((nx, ny), To)
        Told = np.full(N, To)
        
        # Inicializando as Matrizes
        A = np.zeros((N, N))
        B = np.zeros(N)
        tempo = 0
        h = 0
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx, ny)
        
        while tempo < tempo_maximo:
            h += 1
        
            # Parte central
            for i in range(1, ny - 1):
                for m in range(i * nx + 1, (i + 1) * nx - 1):
                    Ap = 1 + 2 * rxt + 2 * ryt
                    Aw = -rxt
                    Ae = -rxt
                    As = -ryt
                    An = -ryt
                    S = Told[m]
        
                    A[m, m] = Ap
                    A[m, m - 1] = Aw
                    A[m, m + 1] = Ae
                    A[m, m + nx] = As
                    A[m, m - nx] = An
                    B[m] = S
        
            nxl = nx / L
            p = nxl * (nx + ny) + nxl * 2
            l = p + nxl - 1
            for b in range(0, int(nxl)):
                for j in range(int(p) + (b * nx), int(l) + (b * nx) + 1):
                    B[j] = Told[j] + q
        
            # Canto Superior Esquerdo
            m = 0
            Ap = 1 + 4 * rxt + 4 * ryt
            Ae = -4 / 3 * rxt
            As = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Tn * ryt
        
            A[m, m] = Ap
            A[m, m + 1] = Ae
            A[m, m + nx] = As
            B[m] = S
        
            # Fronteira Norte
            for m in range(1, nx - 1):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -rxt
                Ae = -rxt
                As = -4 / 3 * ryt
                S = Told[m] + 8 / 3 * Tn * ryt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m + 1] = Ae
                A[m, m + nx] = As
                B[m] = S
        
            # Canto Superior Direito
            m = nx - 1
            Ap = 1 + 4 * rxt + 4 * ryt
            Aw = -4 / 3 * rxt
            As = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Tn * ryt
        
            A[m, m] = Ap
            A[m, m - 1] = Aw
            A[m, m + nx] = As
            B[m] = S
        
            # Fronteira Oeste
            for m in range(nx, (ny - 2) * nx + 1, nx):
                Ap = 1 + 2 * rxt + 4 * ryt
                Ae = -4 / 3 * rxt
                An = -ryt
                As = -ryt
                S = Told[m] + 8 / 3 * Tw * rxt
        
                A[m, m] = Ap
                A[m, m + 1] = Ae
                A[m, m + nx] = As
                A[m, m - nx] = An
                B[m] = S
        
            # Fronteira Leste
            for m in range(2 * nx - 1, (ny - 1) * nx, nx):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -4 / 3 * rxt
                An = -ryt
                As = -ryt
                S = Told[m] + 8 / 3 * Te * rxt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m - nx] = An
                A[m, m + nx] = As
                B[m] = S
        
            # Canto Inferior Esquerdo
            m = (ny - 1) * nx
            Ap = 1 + 4 * rxt + 4 * ryt
            Ae = -4 / 3 * rxt
            An = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Ts * ryt
        
            A[m, m] = Ap
            A[m, m + 1] = Ae
            A[m, m - nx] = An
            B[m] = S
        
            # Fronteira Sul
            for m in range((ny - 1) * nx + 1, ny * nx - 1):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -rxt
                Ae = -rxt
                An = -4 / 3 * ryt
                S = Told[m] + 8 / 3 * Ts * ryt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m + 1] = Ae
                A[m, m - nx] = An
                B[m] = S
        
            # Canto Inferior Direito
            m = ny * nx - 1
            Ap = 1 + 4 * rxt + 4 * ryt
            Aw = -4 / 3 * rxt
            An = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Ts * ryt
        
            A[m, m] = Ap
            A[m, m - 1] = Aw
            A[m, m - nx] = An
            B[m] = S
        
            # Solução do sistema Linear
            x0 = np.ones(N)
            T = solvers.jacobi(A, B, x0, Eppara, maxit)
        
            T_new = np.zeros((nx, ny), dtype=np.float64)
            # Extração do valor da variável T
            for i in range(nx):
                for j in range(ny):
                    T_new[i, j] = T[ind[i, j]]
            
            T_new_inv = T_new[::-1]
        
            Told = T.copy()
            T_old = T_new.copy()
            tempo += dt
        
        # Plot
        plt.figure()
        plt.imshow(T_new_inv, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'Distribuição de Temperatura - Termo Fonte - Jacobi - nx={nx}')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new_inv.T, cmap='viridis')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Temperatura [°C]')
        plt.title(f'Superfície de Temperatura - Termo Fonte - Jacobi - nx={nx}')
        
        plt.show()
        
        temp_simu = time.time() - inicio
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return T_new_inv, temp_simu
        
    def calculate_gsr(nx):
    
        # Critério de Scarvorought, 1966
        n = 12  # Números de algarismos significativos
        Eppara = 0.5 * 10 ** (2 - n)  # Termo relativos
        Lambda = 1.5
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 5  # cm
        Ly = 5  # cm
        L = Lx
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 0  # ºC
        Tn = 0 # ºC
        Ts = 20  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        q = 20000
        
        # Parâmetros de simulação
        nx = nx
        ny = nx
        N = nx * ny
        nt = 10
        
        # Cálculos Iniciais
        alpha = k / (rho * cp)
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)
        dt = tempo_maximo / nt
        rxt = alpha * dt / dx ** 2
        ryt = alpha * dt / dy ** 2
        
        
        # Solução da Equação do Calor - MDFI
        inicio = time.time()
        
        # Condição Inicial
        T_old = np.full((nx, ny), To)
        T_new = np.full((nx, ny), To)
        Told = np.full(N, To)
        
        # Inicializando as Matrizes
        A = np.zeros((N, N))
        B = np.zeros(N)
        tempo = 0
        h = 0
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx, ny)
        
        while tempo < tempo_maximo:
            h += 1
        
            # Parte central
            for i in range(1, ny - 1):
                for m in range(i * nx + 1, (i + 1) * nx - 1):
                    Ap = 1 + 2 * rxt + 2 * ryt
                    Aw = -rxt
                    Ae = -rxt
                    As = -ryt
                    An = -ryt
                    S = Told[m]
        
                    A[m, m] = Ap
                    A[m, m - 1] = Aw
                    A[m, m + 1] = Ae
                    A[m, m + nx] = As
                    A[m, m - nx] = An
                    B[m] = S
        
            nxl = nx / L
            p = nxl * (nx + ny) + nxl * 2
            l = p + nxl - 1
            for b in range(0, int(nxl)):
                for j in range(int(p) + (b * nx), int(l) + (b * nx) + 1):
                    B[j] = Told[j] + q
        
            # Canto Superior Esquerdo
            m = 0
            Ap = 1 + 4 * rxt + 4 * ryt
            Ae = -4 / 3 * rxt
            As = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Tn * ryt
        
            A[m, m] = Ap
            A[m, m + 1] = Ae
            A[m, m + nx] = As
            B[m] = S
        
            # Fronteira Norte
            for m in range(1, nx - 1):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -rxt
                Ae = -rxt
                As = -4 / 3 * ryt
                S = Told[m] + 8 / 3 * Tn * ryt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m + 1] = Ae
                A[m, m + nx] = As
                B[m] = S
        
            # Canto Superior Direito
            m = nx - 1
            Ap = 1 + 4 * rxt + 4 * ryt
            Aw = -4 / 3 * rxt
            As = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Tn * ryt
        
            A[m, m] = Ap
            A[m, m - 1] = Aw
            A[m, m + nx] = As
            B[m] = S
        
            # Fronteira Oeste
            for m in range(nx, (ny - 2) * nx + 1, nx):
                Ap = 1 + 2 * rxt + 4 * ryt
                Ae = -4 / 3 * rxt
                An = -ryt
                As = -ryt
                S = Told[m] + 8 / 3 * Tw * rxt
        
                A[m, m] = Ap
                A[m, m + 1] = Ae
                A[m, m + nx] = As
                A[m, m - nx] = An
                B[m] = S
        
            # Fronteira Leste
            for m in range(2 * nx - 1, (ny - 1) * nx, nx):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -4 / 3 * rxt
                An = -ryt
                As = -ryt
                S = Told[m] + 8 / 3 * Te * rxt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m - nx] = An
                A[m, m + nx] = As
                B[m] = S
        
            # Canto Inferior Esquerdo
            m = (ny - 1) * nx
            Ap = 1 + 4 * rxt + 4 * ryt
            Ae = -4 / 3 * rxt
            An = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Ts * ryt
        
            A[m, m] = Ap
            A[m, m + 1] = Ae
            A[m, m - nx] = An
            B[m] = S
        
            # Fronteira Sul
            for m in range((ny - 1) * nx + 1, ny * nx - 1):
                Ap = 1 + 2 * rxt + 4 * ryt
                Aw = -rxt
                Ae = -rxt
                An = -4 / 3 * ryt
                S = Told[m] + 8 / 3 * Ts * ryt
        
                A[m, m] = Ap
                A[m, m - 1] = Aw
                A[m, m + 1] = Ae
                A[m, m - nx] = An
                B[m] = S
        
            # Canto Inferior Direito
            m = ny * nx - 1
            Ap = 1 + 4 * rxt + 4 * ryt
            Aw = -4 / 3 * rxt
            An = -4 / 3 * ryt
            S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Ts * ryt
        
            A[m, m] = Ap
            A[m, m - 1] = Aw
            A[m, m - nx] = An
            B[m] = S
        
            # Solução do sistema Linear
            x0 = np.ones(N)
            T = solvers.gauss_seidel_relax(A, B, x0, Eppara, maxit, Lambda)
        
            T_new = np.zeros((nx, ny), dtype=np.float64)
            # Extração do valor da variável T
            for i in range(nx):
                for j in range(ny):
                    T_new[i, j] = T[ind[i, j]]
            
            T_new_inv = T_new[::-1]
        
            Told = T.copy()
            T_old = T_new.copy()
            tempo += dt
        
        # Plot
        plt.figure()
        plt.imshow(T_new_inv, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'Distribuição de Temperatura - Termo Fonte - GS Relaxamento - nx={nx}')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new_inv.T, cmap='viridis')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Temperatura [°C]')
        plt.title(f'Superfície de Temperatura - Termo Fonte - GS Relaxamento - nx={nx}')

        
        plt.show()
        
        temp_simu = time.time() - inicio
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return T_new_inv, temp_simu