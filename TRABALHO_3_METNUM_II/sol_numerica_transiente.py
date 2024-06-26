class transiente():
    
    def calculate_transiente_nx10_gs():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 1  # cm
        Ly = 1  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 10
        ny = 10
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - GS - nx=10')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GS - nx=10')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu
        
    def calculate_transiente_nx20_gs():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - GS - nx=20')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GS - nx=20')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nx30_gs():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 30
        ny = 30
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')
        plt.title('Distribuição de Temperatura - ST - GS - nx=30')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Temperatura [°C]')
        plt.title('Superfície de Temperatura - ST - GS - nx=30')
        
        plt.show()
        
        temp_simu = time.time() - inicio
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nx10_jac():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 10
        ny = 10
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.jacobi(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - Jacobi - nx=10')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - Jacobi - nx=10')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu
        
    def calculate_transiente_nx20_jac():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - Jacobi - nx=20')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - Jacobi - nx=20')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nx30_jac():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 30
        ny = 30
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.jacobi(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - Jacobi - nx=30')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - Jacobi - nx=30')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        return temp_simu

    def calculate_transiente_nx10_gsr():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        Lambda = 1.5
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 10
        ny = 10
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel_relax(A, B, x0, Eppara, maxit, Lambda)
        
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
        plt.title('Distribuição de Temperatura - ST - GSR - nx=10')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GSR - nx=10')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu
        
    def calculate_transiente_nx20_gsr():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        Lambda = 1.5
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel_relax(A, B, x0, Eppara, maxit, Lambda)
        
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
        plt.title('Distribuição de Temperatura - ST - GSR - nx=20')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GSR - nx=20')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nx30_gsr():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        Lambda = 1.5
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 30
        ny = 30
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel_relax(A, B, x0, Eppara, maxit, Lambda)
        
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
        plt.title('Distribuição de Temperatura - ST - GSR - nx=30')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GSR - nx=30')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    
    def calculate_transiente_nt10_gs():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 10
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - GS - nx=10')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GS - nx=10')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu
        
    def calculate_transiente_nt50_gs():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 50
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - GS - nx=20')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GS - nx=20')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nt100_gs():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')
        plt.title('Distribuição de Temperatura - ST - GS - nx=30')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Temperatura [°C]')
        plt.title('Superfície de Temperatura - ST - GS - nx=30')
        
        plt.show()
        
        temp_simu = time.time() - inicio
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nt10_jac():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 10
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.jacobi(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - Jacobi - nx=10')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - Jacobi - nx=10')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu
        
    def calculate_transiente_nt50_jac():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 50
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - Jacobi - nx=20')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - Jacobi - nx=20')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nt100_jac():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.jacobi(A, B, x0, Eppara, maxit)
        
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
        plt.title('Distribuição de Temperatura - ST - Jacobi - nx=30')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - Jacobi - nx=30')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        return temp_simu

    def calculate_transiente_nt10_gsr():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        Lambda = 1.5
        
        # Número Máximo de Iterações
        maxit = 1000
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 10
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel_relax(A, B, x0, Eppara, maxit, Lambda)
        
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
        plt.title('Distribuição de Temperatura - ST - GSR - nx=10')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GSR - nx=10')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu
        
    def calculate_transiente_nt50_gsr():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        Lambda = 1.5
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 50
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel_relax(A, B, x0, Eppara, maxit, Lambda)
        
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
        plt.title('Distribuição de Temperatura - ST - GSR - nx=20')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GSR - nx=20')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu

    def calculate_transiente_nt100_gsr():
        
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from solvers import solvers 
        
        
        # Critério de Scarvorought, 1966
        n = 12 # Números de algarismos significativos
        Eppara = 0.5*10**(2-n) # Termo relativos
        
        # Número Máximo de Iterações
        maxit = 1000
        Lambda = 1.5
        
        # Propriedades do Material - Cobre
        rho = 8.92  # g/cm^3
        cp = 0.092  # cal/(g.ºC)
        k = 0.95  # cal/(cm.s.ºC)
        Lx = 20  # cm
        Ly = 10  # cm
        
        # Dados Iniciais
        tempo_maximo = 1000  # segundos
        To = 10  # ºC
        Tn = 10  # ºC
        Ts = 10  # ºC
        Tw = 10  # ºC
        Te = 10  # ºC
        h_flux = 15
        T_inf = 25
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx * ny
        nt = 100
        
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
              Ap = 1 + (4/3)*rxt + 4/3*rxt*((dx*h_flux)/k) + 2*ryt
              Ae = -4/3*rxt
              An = -ryt
              As = -ryt
              S = Told[m] + 4/3*rxt*((dx*h_flux*T_inf)/k)
        
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
            T = solvers.gauss_seidel_relax(A, B, x0, Eppara, maxit, Lambda)
        
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
        plt.title('Distribuição de Temperatura - ST - GSR - nx=30')
        
        plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
        ax.plot_surface(X, Y, T_new.T, cmap='viridis')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Temperatura (°C)')
        plt.title('Superfície de Temperatura - ST - GSR - nx=30')
        
        plt.show()
        
        temp_simu = time.time() - inicio 
        
        print(f'Tempo de simulação: {time.time() - inicio:.2f} segundos')
        
        return temp_simu