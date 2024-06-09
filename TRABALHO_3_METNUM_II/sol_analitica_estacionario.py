     
class analitica():
    
    def calculate_analitica_nx10():
        
        import numpy as np
        import math
        import matplotlib.pyplot as plt
        
        # Propriedades do Material - Cobre
        rho = 8.92 # g/cm^3
        cp = 0.092 # cal/(g.ºC)
        k = 0.95 # cal/(cm.s.ºC)
        Lx = 1 # cm
        Ly = 1 # cm
        N = 200 # Números de termos da Série de Fourier
        
        nx = 10
        ny = 10
        
        # Cálculos Iniciais
        c = k/(rho*cp)
        
        # Define a função desejada:
        def f(x):
          return 20
        
        # Coeficiente da Série de Fourier
        def fourier(f, Lx, Ly, N):
          Bn = np.zeros(N)
        
          for n in range(1, N):
        
            def integrando(x):
              return f(x)*np.sin(n*np.pi*x/Lx)
        
            Bn[n] = 2/(Lx*np.sinh(n*np.pi*Lx/Ly))*np.trapz(integrando(np.linspace(0,Lx,1000)),np.linspace(0,Lx,1000))
        
          return Bn
        
        # Solução da Equação do Calor:
        def calor(x, y, Lx, Ly, N):
          solucao = np.zeros_like(x) # np.zeros: cria uma matriz de zeros com o mesmo formato e tipo de dados de uma matriz de entrada fornecida.
          coef = fourier(f, Lx, Ly, N)
        
          for n in range(1,N):
        
            solucao += coef[n]*np.sin(n*np.pi*x/Lx)*np.sinh(n*np.pi*y/Ly)
        
          return solucao
        
        # Define os valores de x e t
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x,y)
        
        Temperatura = calor(X, Y, Lx, Ly, N)
        
        # Gráfico 1
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Temperatura, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Analítica - nx=10')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        
        return Temperatura, nx, X, Y 
        
    def calculate_analitica_nx20():
        
        import numpy as np
        import math
        import matplotlib.pyplot as plt
        
        # Propriedades do Material - Cobre
        rho = 8.92 # g/cm^3
        cp = 0.092 # cal/(g.ºC)
        k = 0.95 # cal/(cm.s.ºC)
        Lx = 10 # cm
        Ly = 10 # cm
        N = 200 # Números de termos da Série de Fourier
        
        nx = 20
        ny = 20
        
        # Cálculos Iniciais
        c = k/(rho*cp)
        
        # Define a função desejada:
        def f(x):
          return 20
        
        # Coeficiente da Série de Fourier
        def fourier(f, Lx, Ly, N):
          Bn = np.zeros(N)
        
          for n in range(1, N):
        
            def integrando(x):
              return f(x)*np.sin(n*np.pi*x/Lx)
        
            Bn[n] = 2/(Lx*np.sinh(n*np.pi*Lx/Ly))*np.trapz(integrando(np.linspace(0,Lx,1000)),np.linspace(0,Lx,1000))
        
          return Bn
        
        # Solução da Equação do Calor:
        def calor(x, y, Lx, Ly, N):
          solucao = np.zeros_like(x) # np.zeros: cria uma matriz de zeros com o mesmo formato e tipo de dados de uma matriz de entrada fornecida.
          coef = fourier(f, Lx, Ly, N)
        
          for n in range(1,N):
        
            solucao += coef[n]*np.sin(n*np.pi*x/Lx)*np.sinh(n*np.pi*y/Ly)
        
          return solucao
        
        # Define os valores de x e t
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x,y)
        
        Temperatura = calor(X, Y, Lx, Ly, N)
        
        # Gráfico 1
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Temperatura, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Analítica - nx=20')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        return Temperatura, nx, X, Y  
        
    def calculate_analitica_nx30():
        
        import numpy as np
        import math
        import matplotlib.pyplot as plt
        
        # Propriedades do Material - Cobre
        rho = 8.92 # g/cm^3
        cp = 0.092 # cal/(g.ºC)
        k = 0.95 # cal/(cm.s.ºC)
        Lx = 10 # cm
        Ly = 10 # cm
        N = 200 # Números de termos da Série de Fourier
        
        nx = 30
        ny = 30
        
        # Cálculos Iniciais
        c = k/(rho*cp)
        
        # Define a função desejada:
        def f(x):
          return 20
        
        # Coeficiente da Série de Fourier
        def fourier(f, Lx, Ly, N):
          Bn = np.zeros(N)
        
          for n in range(1, N):
        
            def integrando(x):
              return f(x)*np.sin(n*np.pi*x/Lx)
        
            Bn[n] = 2/(Lx*np.sinh(n*np.pi*Lx/Ly))*np.trapz(integrando(np.linspace(0,Lx,1000)),np.linspace(0,Lx,1000))
        
          return Bn
        
        # Solução da Equação do Calor:
        def calor(x, y, Lx, Ly, N):
          solucao = np.zeros_like(x) # np.zeros: cria uma matriz de zeros com o mesmo formato e tipo de dados de uma matriz de entrada fornecida.
          coef = fourier(f, Lx, Ly, N)
        
          for n in range(1,N):
        
            solucao += coef[n]*np.sin(n*np.pi*x/Lx)*np.sinh(n*np.pi*y/Ly)
        
          return solucao
        
        # Define os valores de x e t
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x,y)
        
        Temperatura = calor(X, Y, Lx, Ly, N)
        
        # Gráfico 1
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Temperatura, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Analítica - nx=30')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        return Temperatura, nx, X, Y  

    def calculate_analitica_ny10():
        
        import numpy as np
        import math
        import matplotlib.pyplot as plt
        
        # Propriedades do Material - Cobre
        rho = 8.92 # g/cm^3
        cp = 0.092 # cal/(g.ºC)
        k = 0.95 # cal/(cm.s.ºC)
        Lx = 1 # cm
        Ly = 1 # cm
        N = 200 # Números de termos da Série de Fourier
        
        nx = 10
        ny = 10
        
        # Cálculos Iniciais
        c = k/(rho*cp)
        
        # Define a função desejada:
        def f(x):
          return 20
        
        # Coeficiente da Série de Fourier
        def fourier(f, Lx, Ly, N):
          Bn = np.zeros(N)
        
          for n in range(1, N):
        
            def integrando(x):
              return f(x)*np.sin(n*np.pi*x/Lx)
        
            Bn[n] = 2/(Lx*np.sinh(n*np.pi*Lx/Ly))*np.trapz(integrando(np.linspace(0,Lx,1000)),np.linspace(0,Lx,1000))
        
          return Bn
        
        # Solução da Equação do Calor:
        def calor(x, y, Lx, Ly, N):
          solucao = np.zeros_like(x) # np.zeros: cria uma matriz de zeros com o mesmo formato e tipo de dados de uma matriz de entrada fornecida.
          coef = fourier(f, Lx, Ly, N)
        
          for n in range(1,N):
        
            solucao += coef[n]*np.sin(n*np.pi*x/Lx)*np.sinh(n*np.pi*y/Ly)
        
          return solucao
        
        # Define os valores de x e t
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x,y)
        
        Temperatura = calor(X, Y, Lx, Ly, N)
        
        # Gráfico 1
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Temperatura, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Analítica - ny=10')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        
        return Temperatura, ny, X, Y 
        
    def calculate_analitica_ny20():
        
        import numpy as np
        import math
        import matplotlib.pyplot as plt
        
        # Propriedades do Material - Cobre
        rho = 8.92 # g/cm^3
        cp = 0.092 # cal/(g.ºC)
        k = 0.95 # cal/(cm.s.ºC)
        Lx = 1 # cm
        Ly = 1 # cm
        N = 200 # Números de termos da Série de Fourier
        
        nx = 20
        ny = 20
        
        # Cálculos Iniciais
        c = k/(rho*cp)
        
        # Define a função desejada:
        def f(x):
          return 20
        
        # Coeficiente da Série de Fourier
        def fourier(f, Lx, Ly, N):
          Bn = np.zeros(N)
        
          for n in range(1, N):
        
            def integrando(x):
              return f(x)*np.sin(n*np.pi*x/Lx)
        
            Bn[n] = 2/(Lx*np.sinh(n*np.pi*Lx/Ly))*np.trapz(integrando(np.linspace(0,Lx,1000)),np.linspace(0,Lx,1000))
        
          return Bn
        
        # Solução da Equação do Calor:
        def calor(x, y, Lx, Ly, N):
          solucao = np.zeros_like(x) # np.zeros: cria uma matriz de zeros com o mesmo formato e tipo de dados de uma matriz de entrada fornecida.
          coef = fourier(f, Lx, Ly, N)
        
          for n in range(1,N):
        
            solucao += coef[n]*np.sin(n*np.pi*x/Lx)*np.sinh(n*np.pi*y/Ly)
        
          return solucao
        
        # Define os valores de x e t
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x,y)
        
        Temperatura = calor(X, Y, Lx, Ly, N)
        
        # Gráfico 1
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Temperatura, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Analítica - ny=20')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        
        return Temperatura, ny, X, Y  

    def calculate_analitica_ny30():
        
        import numpy as np
        import math
        import matplotlib.pyplot as plt
        
        # Propriedades do Material - Cobre
        rho = 8.92 # g/cm^3
        cp = 0.092 # cal/(g.ºC)
        k = 0.95 # cal/(cm.s.ºC)
        Lx = 1 # cm
        Ly = 1 # cm
        N = 200 # Números de termos da Série de Fourier
        
        nx = 30
        ny = 30
        
        # Cálculos Iniciais
        c = k/(rho*cp)
        
        # Define a função desejada:
        def f(x):
          return 20
        
        # Coeficiente da Série de Fourier
        def fourier(f, Lx, Ly, N):
          Bn = np.zeros(N)
        
          for n in range(1, N):
        
            def integrando(x):
              return f(x)*np.sin(n*np.pi*x/Lx)
        
            Bn[n] = 2/(Lx*np.sinh(n*np.pi*Lx/Ly))*np.trapz(integrando(np.linspace(0,Lx,1000)),np.linspace(0,Lx,1000))
        
          return Bn
        
        # Solução da Equação do Calor:
        def calor(x, y, Lx, Ly, N):
          solucao = np.zeros_like(x) # np.zeros: cria uma matriz de zeros com o mesmo formato e tipo de dados de uma matriz de entrada fornecida.
          coef = fourier(f, Lx, Ly, N)
        
          for n in range(1,N):
        
            solucao += coef[n]*np.sin(n*np.pi*x/Lx)*np.sinh(n*np.pi*y/Ly)
        
          return solucao
        
        # Define os valores de x e t
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x,y)
        
        Temperatura = calor(X, Y, Lx, Ly, N)
        
        # Gráfico 1
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Temperatura, rstride=2, cstride=1, cmap=plt.cm.viridis, linewidth=0.2, alpha=1)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('T(x,y) [°C]')
        plt.title('Equação do Calor 2D - Solução Analítica - ny=30')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        #ax.view_init(30, 30)
        
        return Temperatura, ny, X, Y 
