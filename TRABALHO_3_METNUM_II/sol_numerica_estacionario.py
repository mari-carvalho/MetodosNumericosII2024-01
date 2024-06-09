class estacionario():
    
    def calculate_estacionario_nx10():

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
        Lx = 1  # cm
        Ly = 1  # cm
        
        # Dados Iniciais
        Tn = 20  # ºC
        Ts = 0  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        
        # Parâmetros de simulação
        nx = 10
        ny = 10
        N = nx*ny
        
        # Cálculos Iniciais
        alpha = k/(rho*cp)
        dx = Lx/nx
        dy = Ly/ny
        rxt = alpha/dx**2
        ryt = alpha/dy**2
        
        rxt = dy**2
        ryt = dx**2
        
        # Matriz Solução
        TS = np.full((nx,ny),0)
        A = np.zeros((N,N))
        B = np.zeros(N)
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx,ny)
        
        # Parte central
        for i in range(1,ny-1):
          for m in range(i*nx+1,(i+1)*nx-1):
            Ap = 2*rxt + 2*ryt
            Aw = -rxt
            Ae = -rxt
            As = -ryt
            An = -ryt
            S = 0
        
            A[m,m] = Ap
            A[m,m-1] = Aw
            A[m,m+1] = Ae
            A[m,m+nx] = As
            A[m,m-nx] = An
            B[m] = S
        
        # Canto Superior Esquerdo
        m = 0
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Norte
        for m in range(1,nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          As = -4/3*ryt
          S = 8/3*Tn*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Superior Direito
        m = nx-1
        Ap = 4*rxt + 4*ryt
        Aw = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Oeste
        for m in range(nx,(ny-2)*nx+1,nx):
          Ap = 2*rxt + 4*ryt
          Ae = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Tw*rxt
        
          A[m,m] = Ap
          A[m,m+1] = Ae
          A[m,m+nx] = As
          A[m,m-nx] = An
          B[m] = S
        
        # Fronteira Leste
        for m in range(2*nx-1,(ny-1)*nx,nx):
          Ap = 2*rxt + 4*ryt
          Aw = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Te*rxt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m-nx] = An
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Inferior Esquerdo
        m = (ny-1)*nx
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m-nx] = An
        B[m] = S
        
        # Fronteira Sul
        for m in range((ny-1)*nx+1,ny*nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          An = -4/3*ryt
          S = 8/3*Ts*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m-nx] = An
          B[m] = S
        
        # Canto Inferior Direito
        m = ny*nx-1
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m-nx] = An
        B[m] = S
        
        # Solução do sistema Linear
        x0 = np.ones(N)
        T = Gauss_Seidel(A, B, x0, Eppara, maxit)
        #T = np.linalg.solve(A, B)
        
        TS = np.zeros((nx, ny), dtype=np.float64)
        # Extração do valor da variável T
        for i in range(nx):
            for j in range(ny):
                TS[i, j] = T[ind[i, j]]
                
        TS_plot = TS[::-1]

        
        # Plot
        plt.figure()
        plt.imshow(TS_plot, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        plt.title('Distribuição de Temperatura - Solução Estacionário - nx=10')
        
        # Plotagem
        X = np.linspace(0, Lx, nx)
        Y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(X, Y)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, TS_plot, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('T(x,y) (°C)')
        plt.title('Equação do Calor 2D - Solução Estacionário - nx=10')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, -30)
        plt.show()
        return TS, nx, X, Y 
    
    def calculate_estacionario_nx20():
    
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
        Lx = 1  # cm
        Ly = 1  # cm
        
        # Dados Iniciais
        Tn = 20  # ºC
        Ts = 0  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx*ny
        
        # Cálculos Iniciais
        alpha = k/(rho*cp)
        dx = Lx/nx
        dy = Ly/ny
        rxt = alpha/dx**2
        ryt = alpha/dy**2
        
        rxt = dy**2
        ryt = dx**2
        
        # Matriz Solução
        TS = np.full((nx,ny),0)
        A = np.zeros((N,N))
        B = np.zeros(N)
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx,ny)
        
        # Parte central
        for i in range(1,ny-1):
          for m in range(i*nx+1,(i+1)*nx-1):
            Ap = 2*rxt + 2*ryt
            Aw = -rxt
            Ae = -rxt
            As = -ryt
            An = -ryt
            S = 0
        
            A[m,m] = Ap
            A[m,m-1] = Aw
            A[m,m+1] = Ae
            A[m,m+nx] = As
            A[m,m-nx] = An
            B[m] = S
        
        # Canto Superior Esquerdo
        m = 0
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Norte
        for m in range(1,nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          As = -4/3*ryt
          S = 8/3*Tn*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Superior Direito
        m = nx-1
        Ap = 4*rxt + 4*ryt
        Aw = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Oeste
        for m in range(nx,(ny-2)*nx+1,nx):
          Ap = 2*rxt + 4*ryt
          Ae = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Tw*rxt
        
          A[m,m] = Ap
          A[m,m+1] = Ae
          A[m,m+nx] = As
          A[m,m-nx] = An
          B[m] = S
        
        # Fronteira Leste
        for m in range(2*nx-1,(ny-1)*nx,nx):
          Ap = 2*rxt + 4*ryt
          Aw = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Te*rxt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m-nx] = An
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Inferior Esquerdo
        m = (ny-1)*nx
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m-nx] = An
        B[m] = S
        
        # Fronteira Sul
        for m in range((ny-1)*nx+1,ny*nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          An = -4/3*ryt
          S = 8/3*Ts*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m-nx] = An
          B[m] = S
        
        # Canto Inferior Direito
        m = ny*nx-1
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m-nx] = An
        B[m] = S
        
        # Solução do sistema Linear
        x0 = np.ones(N)
        T = Gauss_Seidel(A, B, x0, Eppara, maxit)
        #T = np.linalg.solve(A, B)
        
        TS = np.zeros((nx, ny), dtype=np.float64)
        # Extração do valor da variável T
        for i in range(nx):
            for j in range(ny):
                TS[i, j] = T[ind[i, j]]
                
        TS_plot = TS[::-1]
        
        # Plot
        plt.figure()
        plt.imshow(TS_plot, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        plt.title('Distribuição de Temperatura - Solução Estacionário - nx=20')
        
        # Plotagem
        X = np.linspace(0, Lx, nx)
        Y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(X, Y)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, TS_plot, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('T(x,y) (°C)')
        plt.title('Equação do Calor 2D - Solução Estacionário - nx=20')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        plt.show()
        return TS, nx, X, Y 
        
    def calculate_estacionario_nx30():
    
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
        Lx = 1  # cm
        Ly = 1  # cm
        
        # Dados Iniciais
        Tn = 20  # ºC
        Ts = 0  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        
        # Parâmetros de simulação
        nx = 30
        ny = 30
        N = nx*ny
        
        # Cálculos Iniciais
        alpha = k/(rho*cp)
        dx = Lx/nx
        dy = Ly/ny
        rxt = alpha/dx**2
        ryt = alpha/dy**2
        
        rxt = dy**2
        ryt = dx**2
        
        # Matriz Solução
        TS = np.full((nx,ny),0)
        A = np.zeros((N,N))
        B = np.zeros(N)
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx,ny)
        
        # Parte central
        for i in range(1,ny-1):
          for m in range(i*nx+1,(i+1)*nx-1):
            Ap = 2*rxt + 2*ryt
            Aw = -rxt
            Ae = -rxt
            As = -ryt
            An = -ryt
            S = 0
        
            A[m,m] = Ap
            A[m,m-1] = Aw
            A[m,m+1] = Ae
            A[m,m+nx] = As
            A[m,m-nx] = An
            B[m] = S
        
        # Canto Superior Esquerdo
        m = 0
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Norte
        for m in range(1,nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          As = -4/3*ryt
          S = 8/3*Tn*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Superior Direito
        m = nx-1
        Ap = 4*rxt + 4*ryt
        Aw = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Oeste
        for m in range(nx,(ny-2)*nx+1,nx):
          Ap = 2*rxt + 4*ryt
          Ae = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Tw*rxt
        
          A[m,m] = Ap
          A[m,m+1] = Ae
          A[m,m+nx] = As
          A[m,m-nx] = An
          B[m] = S
        
        # Fronteira Leste
        for m in range(2*nx-1,(ny-1)*nx,nx):
          Ap = 2*rxt + 4*ryt
          Aw = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Te*rxt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m-nx] = An
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Inferior Esquerdo
        m = (ny-1)*nx
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m-nx] = An
        B[m] = S
        
        # Fronteira Sul
        for m in range((ny-1)*nx+1,ny*nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          An = -4/3*ryt
          S = 8/3*Ts*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m-nx] = An
          B[m] = S
        
        # Canto Inferior Direito
        m = ny*nx-1
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m-nx] = An
        B[m] = S
        
        # Solução do sistema Linear
        x0 = np.ones(N)
        T = Gauss_Seidel(A, B, x0, Eppara, maxit)
        #T = np.linalg.solve(A, B)
        
        TS = np.zeros((nx, ny), dtype=np.float64)
        # Extração do valor da variável T
        for i in range(nx):
            for j in range(ny):
                TS[i, j] = T[ind[i, j]]
        
        TS_plot = TS[::-1]       
        
        
        # Plot
        plt.figure()
        plt.imshow(TS_plot, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        plt.title('Distribuição de Temperatura - Solução Estacionário - nx=30')
        
        # Plotagem
        X = np.linspace(0, Lx, nx)
        Y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(X, Y)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, TS_plot, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('T(x,y) (°C)')
        plt.title('Equação do Calor 2D - Solução Estacionário - nx=30')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        plt.show()
        return TS, nx, X, Y 
    
    def calculate_estacionario_ny10():

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
        Lx = 1  # cm
        Ly = 1  # cm
        
        # Dados Iniciais
        Tn = 20  # ºC
        Ts = 0  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        
        # Parâmetros de simulação
        nx = 10
        ny = 10
        N = nx*ny
        
        # Cálculos Iniciais
        alpha = k/(rho*cp)
        dx = Lx/nx
        dy = Ly/ny
        rxt = alpha/dx**2
        ryt = alpha/dy**2
        
        rxt = dy**2
        ryt = dx**2
        
        # Matriz Solução
        TS = np.full((nx,ny),0)
        A = np.zeros((N,N))
        B = np.zeros(N)
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx,ny)
        
        # Parte central
        for i in range(1,ny-1):
          for m in range(i*nx+1,(i+1)*nx-1):
            Ap = 2*rxt + 2*ryt
            Aw = -rxt
            Ae = -rxt
            As = -ryt
            An = -ryt
            S = 0
        
            A[m,m] = Ap
            A[m,m-1] = Aw
            A[m,m+1] = Ae
            A[m,m+nx] = As
            A[m,m-nx] = An
            B[m] = S
        
        # Canto Superior Esquerdo
        m = 0
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Norte
        for m in range(1,nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          As = -4/3*ryt
          S = 8/3*Tn*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Superior Direito
        m = nx-1
        Ap = 4*rxt + 4*ryt
        Aw = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Oeste
        for m in range(nx,(ny-2)*nx+1,nx):
          Ap = 2*rxt + 4*ryt
          Ae = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Tw*rxt
        
          A[m,m] = Ap
          A[m,m+1] = Ae
          A[m,m+nx] = As
          A[m,m-nx] = An
          B[m] = S
        
        # Fronteira Leste
        for m in range(2*nx-1,(ny-1)*nx,nx):
          Ap = 2*rxt + 4*ryt
          Aw = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Te*rxt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m-nx] = An
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Inferior Esquerdo
        m = (ny-1)*nx
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m-nx] = An
        B[m] = S
        
        # Fronteira Sul
        for m in range((ny-1)*nx+1,ny*nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          An = -4/3*ryt
          S = 8/3*Ts*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m-nx] = An
          B[m] = S
        
        # Canto Inferior Direito
        m = ny*nx-1
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m-nx] = An
        B[m] = S
        
        # Solução do sistema Linear
        x0 = np.ones(N)
        T = Gauss_Seidel(A, B, x0, Eppara, maxit)
        #T = np.linalg.solve(A, B)
        
        TS = np.zeros((nx, ny), dtype=np.float64)
        # Extração do valor da variável T
        for i in range(nx):
            for j in range(ny):
                TS[i, j] = T[ind[i, j]]
        
        TS_plot = TS[::-1]
        
        
        # Plot
        plt.figure()
        plt.imshow(TS_plot, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        plt.title('Distribuição de Temperatura - Solução Estacionário - ny=10')
        
        # Plotagem
        X = np.linspace(0, Lx, nx)
        Y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(X, Y)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, TS_plot, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Estacionário - ny=10')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        plt.show()
        return TS, ny, X, Y 
        
    def calculate_estacionario_ny20():

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
        Lx = 1  # cm
        Ly = 1  # cm
        
        # Dados Iniciais
        Tn = 20  # ºC
        Ts = 0  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        
        # Parâmetros de simulação
        nx = 20
        ny = 20
        N = nx*ny
        
        # Cálculos Iniciais
        alpha = k/(rho*cp)
        dx = Lx/nx
        dy = Ly/ny
        rxt = alpha/dx**2
        ryt = alpha/dy**2
        
        rxt = dy**2
        ryt = dx**2
        
        # Matriz Solução
        TS = np.full((nx,ny),0)
        A = np.zeros((N,N))
        B = np.zeros(N)
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx,ny)
        
        # Parte central
        for i in range(1,ny-1):
          for m in range(i*nx+1,(i+1)*nx-1):
            Ap = 2*rxt + 2*ryt
            Aw = -rxt
            Ae = -rxt
            As = -ryt
            An = -ryt
            S = 0
        
            A[m,m] = Ap
            A[m,m-1] = Aw
            A[m,m+1] = Ae
            A[m,m+nx] = As
            A[m,m-nx] = An
            B[m] = S
        
        # Canto Superior Esquerdo
        m = 0
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Norte
        for m in range(1,nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          As = -4/3*ryt
          S = 8/3*Tn*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Superior Direito
        m = nx-1
        Ap = 4*rxt + 4*ryt
        Aw = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Oeste
        for m in range(nx,(ny-2)*nx+1,nx):
          Ap = 2*rxt + 4*ryt
          Ae = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Tw*rxt
        
          A[m,m] = Ap
          A[m,m+1] = Ae
          A[m,m+nx] = As
          A[m,m-nx] = An
          B[m] = S
        
        # Fronteira Leste
        for m in range(2*nx-1,(ny-1)*nx,nx):
          Ap = 2*rxt + 4*ryt
          Aw = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Te*rxt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m-nx] = An
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Inferior Esquerdo
        m = (ny-1)*nx
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m-nx] = An
        B[m] = S
        
        # Fronteira Sul
        for m in range((ny-1)*nx+1,ny*nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          An = -4/3*ryt
          S = 8/3*Ts*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m-nx] = An
          B[m] = S
        
        # Canto Inferior Direito
        m = ny*nx-1
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m-nx] = An
        B[m] = S
        
        # Solução do sistema Linear
        x0 = np.ones(N)
        T = Gauss_Seidel(A, B, x0, Eppara, maxit)
        #T = np.linalg.solve(A, B)
        
        TS = np.zeros((nx, ny), dtype=np.float64)
        # Extração do valor da variável T
        for i in range(nx):
            for j in range(ny):
                TS[i, j] = T[ind[i, j]]
        
        TS_plot = TS[::-1]
        
        # Plot
        plt.figure()
        plt.imshow(TS_plot, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')
        plt.title('Distribuição de Temperatura - Solução Estacionário - ny=20')
        
        # Plotagem
        X = np.linspace(0, Lx, nx)
        Y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(X, Y)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, TS_plot, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Estacionário - ny=20')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        plt.show()
        return TS, ny, X, Y 

    def calculate_estacionario_ny30():

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
        Lx = 1  # cm
        Ly = 1  # cm
        
        # Dados Iniciais
        Tn = 20  # ºC
        Ts = 0  # ºC
        Tw = 0  # ºC
        Te = 0  # ºC
        
        # Parâmetros de simulação
        nx = 30
        ny = 30
        N = nx*ny
        
        # Cálculos Iniciais
        alpha = k/(rho*cp)
        dx = Lx/nx
        dy = Ly/ny
        rxt = alpha/dx**2
        ryt = alpha/dy**2
        
        rxt = dy**2
        ryt = dx**2
        
        # Matriz Solução
        TS = np.full((nx,ny),0)
        A = np.zeros((N,N))
        B = np.zeros(N)
        
        # Definindo os índices da matriz 2D para o vetor 1D
        ind = np.arange(N).reshape(nx,ny)
        
        # Parte central
        for i in range(1,ny-1):
          for m in range(i*nx+1,(i+1)*nx-1):
            Ap = 2*rxt + 2*ryt
            Aw = -rxt
            Ae = -rxt
            As = -ryt
            An = -ryt
            S = 0
        
            A[m,m] = Ap
            A[m,m-1] = Aw
            A[m,m+1] = Ae
            A[m,m+nx] = As
            A[m,m-nx] = An
            B[m] = S
        
        # Canto Superior Esquerdo
        m = 0
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Norte
        for m in range(1,nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          As = -4/3*ryt
          S = 8/3*Tn*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Superior Direito
        m = nx-1
        Ap = 4*rxt + 4*ryt
        Aw = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Tn*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m+nx] = As
        B[m] = S
        
        # Fronteira Oeste
        for m in range(nx,(ny-2)*nx+1,nx):
          Ap = 2*rxt + 4*ryt
          Ae = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Tw*rxt
        
          A[m,m] = Ap
          A[m,m+1] = Ae
          A[m,m+nx] = As
          A[m,m-nx] = An
          B[m] = S
        
        # Fronteira Leste
        for m in range(2*nx-1,(ny-1)*nx,nx):
          Ap = 2*rxt + 4*ryt
          Aw = -4/3*rxt
          An = -ryt
          As = -ryt
          S = 8/3*Te*rxt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m-nx] = An
          A[m,m+nx] = As
          B[m] = S
        
        # Canto Inferior Esquerdo
        m = (ny-1)*nx
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Tw*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m+1] = Ae
        A[m,m-nx] = An
        B[m] = S
        
        # Fronteira Sul
        for m in range((ny-1)*nx+1,ny*nx-1):
          Ap = 2*rxt + 4*ryt
          Aw = -rxt
          Ae = -rxt
          An = -4/3*ryt
          S = 8/3*Ts*ryt
        
          A[m,m] = Ap
          A[m,m-1] = Aw
          A[m,m+1] = Ae
          A[m,m-nx] = An
          B[m] = S
        
        # Canto Inferior Direito
        m = ny*nx-1
        Ap = 4*rxt + 4*ryt
        Ae = -4/3*rxt
        As = -4/3*ryt
        S = 8/3*Te*rxt + 8/3*Ts*ryt
        
        A[m,m] = Ap
        A[m,m-1] = Aw
        A[m,m-nx] = An
        B[m] = S
        
        # Solução do sistema Linear
        x0 = np.ones(N)
        T = Gauss_Seidel(A, B, x0, Eppara, maxit)
        #T = np.linalg.solve(A, B)
        
        TS = np.zeros((nx, ny), dtype=np.float64)
        # Extração do valor da variável T
        for i in range(nx):
            for j in range(ny):
                TS[i, j] = T[ind[i, j]]
        
        TS_plot = TS[::-1]
        
        
        # Plot
        plt.figure()
        plt.imshow(TS_plot, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar()
        plt.xlabel('X [cm]')
        plt.ylabel('Y [cm]')
        plt.title('Distribuição de Temperatura - Solução Estacionário - ny=30')
        
        # Plotagem
        X = np.linspace(0, Lx, nx)
        Y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(X, Y)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, TS_plot, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Estacionário - ny=30')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        plt.show()
        return TS, ny, X, Y 