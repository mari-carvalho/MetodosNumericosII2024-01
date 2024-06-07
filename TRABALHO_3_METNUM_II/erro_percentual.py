# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:45:18 2024

@author: 03950025081
"""
import matplotlib.pyplot as plt 
from sol_analitica_estacionario import analitica
from sol_numerica_estacionario import estacionario

def calculate_variacao_nx():
    
    # Variando NX:
        
    temperatura_analitica = []
    nx_list_analitico = []
    temperatura_estacionario = []
    nx_list_estacionario = []
    x_list_analitico = []
    y_list_analitico = []
    x_list_numerico = []
    y_list_numerico = []
    
    # Analítica
    
    analitica_nx10, nx, x, y = analitica.calculate_analitica_nx10()
    temperatura_analitica.append(analitica_nx10)
    nx_list_analitico.append(nx)
    x_list_analitico.append(x)
    y_list_analitico.append(y)
    
    analitica_nx20, nx, x, y = analitica.calculate_analitica_nx20()
    temperatura_analitica.append(analitica_nx20)
    nx_list_analitico.append(nx)
    x_list_analitico.append(x)
    y_list_analitico.append(y)
    
    analitica_nx30, nx, x, y = analitica.calculate_analitica_nx30()
    temperatura_analitica.append(analitica_nx30)
    nx_list_analitico.append(nx)
    x_list_analitico.append(x)
    y_list_analitico.append(y)
    
    # Numérico
    
    estacionario_nx10, nx, x, y = estacionario.calculate_estacionario_nx10()
    temperatura_estacionario.append(estacionario_nx10)
    nx_list_estacionario.append(nx)
    x_list_numerico.append(x)
    y_list_numerico.append(y)
    
    estacionario_nx20, nx, x, y = estacionario.calculate_estacionario_nx20()
    temperatura_estacionario.append(estacionario_nx20)
    nx_list_estacionario.append(nx)
    x_list_numerico.append(x)
    y_list_numerico.append(y)
    
    estacionario_nx30, nx, x, y = estacionario.calculate_estacionario_nx30()
    temperatura_estacionario.append(estacionario_nx30)
    nx_list_estacionario.append(nx)
    x_list_numerico.append(x)
    y_list_numerico.append(y)
    
    # Cálculo do Erro 
    
    erro_percentual = []
    sum = 0 
    
    for i in range(len(temperatura_analitica)):
        temp_analitico = temperatura_analitica[i]
        temp_estac = temperatura_estacionario[i]
        if i == 0 :
            ui = temp_analitico[8]
            vi = temp_estac[0]
            nx = nx_list_estacionario[i]
            for j in range(len(ui)-4):
                uii = ui[j+1]
                vii = vi[j]
                sum = sum + abs((uii - vii) / uii)
            erro = 1/nx * sum 
            erro_percentual.append(erro)
        elif i == 1:
            ui = temp_analitico[18]
            vi = temp_estac[0]
            nx = nx_list_estacionario[i]
            for j in range(len(ui)-6):
                uii = ui[j+1]
                vii = vi[j]
                sum = sum + abs((uii - vii) / uii)
            erro = 1/nx * sum 
            erro_percentual.append(erro)
        elif i == 2:
            ui = temp_analitico[28]
            vi = temp_estac[0]
            nx = nx_list_estacionario[i]
            for j in range(len(ui)-6):
                uii = ui[j+1]
                vii = vi[j]
                sum = sum + abs((uii - vii) / uii)
            erro = 1/nx * sum 
            erro_percentual.append(erro)
    return temperatura_analitica, temperatura_estacionario, erro_percentual, nx_list_analitico, nx_list_estacionario, x_list_analitico, y_list_analitico, x_list_numerico, y_list_numerico

def calculate_variacao_ny():
    
    # Variando NY:
        
    temperatura_analitica = []
    ny_list_analitico = []
    temperatura_estacionario = []
    ny_list_estacionario = []
    x_list_analitico = []
    y_list_analitico = []
    x_list_numerico = []
    y_list_numerico = []
    
    # Analítica
    
    analitica_ny10, ny, x, y = analitica.calculate_analitica_ny10()
    temperatura_analitica.append(analitica_ny10)
    ny_list_analitico.append(ny)
    x_list_analitico.append(x)
    y_list_analitico.append(y)

    analitica_ny20, ny, x, y = analitica.calculate_analitica_ny20()
    temperatura_analitica.append(analitica_ny20)
    ny_list_analitico.append(ny)
    x_list_analitico.append(x)
    y_list_analitico.append(y)
    
    analitica_ny30, ny, x, y = analitica.calculate_analitica_ny30()
    temperatura_analitica.append(analitica_ny30)
    ny_list_analitico.append(ny)
    x_list_analitico.append(x)
    y_list_analitico.append(y)
    
    # Numérico
    
    estacionario_ny10, ny, x, y = estacionario.calculate_estacionario_ny10()
    temperatura_estacionario.append(estacionario_ny10)
    ny_list_estacionario.append(ny)
    x_list_numerico.append(x)
    y_list_numerico.append(y)
    
    
    estacionario_ny20, ny, x, y = estacionario.calculate_estacionario_ny20()
    temperatura_estacionario.append(estacionario_ny20)
    ny_list_estacionario.append(ny)
    x_list_numerico.append(x)
    y_list_numerico.append(y)
    
    estacionario_ny30, ny, x, y = estacionario.calculate_estacionario_ny30()
    temperatura_estacionario.append(estacionario_ny30)
    ny_list_estacionario.append(ny)
    x_list_numerico.append(x)
    y_list_numerico.append(y)
    
    # Cálculo do Erro 
    
    erro_percentual = []
    sum = 0 
    
    for i in range(len(temperatura_analitica)):
        temp_analitico = temperatura_analitica[i]
        temp_estac = temperatura_estacionario[i]
        if i == 0 :
            ui = temp_analitico[:,2]
            vi = temp_estac[:,2]
            ny = ny_list_estacionario[i]
            vi_inv = vi[::-1]
            for j in range(1,len(ui)):
                uii = ui[j]
                vii = vi_inv[j]
                sum = sum + abs((uii - vii) / uii)
            erro = 1/ny * sum 
            erro_percentual.append(erro)
        elif i == 1:
            ui = temp_analitico[:, 2]
            vi = temp_estac[:, 2]
            ny = ny_list_estacionario[i]
            vi_inv = vi[::-1]
            for j in range(1,len(ui)):
                uii = ui[j]
                vii = vi_inv[j]
                sum = sum + abs((uii - vii) / uii)
            erro = 1/ny * sum 
            erro_percentual.append(erro)
        elif i == 2:
            ui = temp_analitico[:,2]
            vi = temp_estac[:,2]
            ny = ny_list_estacionario[i]
            vi_inv = vi[::-1]
            for j in range(1,len(ui)):
                uii = ui[j]
                vii = vi_inv[j]
                sum = sum + abs((uii - vii) / uii)
            erro = 1/ny * sum 
            erro_percentual.append(erro)
    return temperatura_analitica, temperatura_estacionario, erro_percentual, ny_list_analitico, ny_list_estacionario, x_list_analitico, y_list_analitico, x_list_numerico, y_list_numerico

temperatura_analitica_nx, tempratura_estacionario_nx, erro_variacao_nx, nx_list_analitico, nx_list_estacionario, x_list_analitico, y_list_analitico, x_list_numerico, y_list_numerico = calculate_variacao_nx()
temperatura_analitica_ny, tempratura_estacionario_ny, erro_variacao_ny, ny_list_analitico, ny_list_estacionario, x_list_analitico, y_list_analitico,  x_list_numerico, y_list_numerico = calculate_variacao_ny()

for i in range(len(temperatura_analitica_nx)):
    analitica2 = temperatura_analitica_nx[i]
    x_analitico = x_list_analitico[i]
    x_a = x_analitico[5]
    y_a = analitica2[5]
    numerico = tempratura_estacionario_nx[i]
    x_numerico = x_list_numerico[i]
    x_n = x_numerico[5]
    y_n = numerico[5]
    plt.plot(x_a, y_a, label='nx={}'.format(nx_list_analitico))
    plt.plot(x_n, y_n, label='nx={}'.format(nx_list_estacionario))

plt.legend()
plt.xlabel('Comprimento [m]')
plt.ylabel('Temperatura [°C]')
plt.show()



    
