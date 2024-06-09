# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:45:18 2024

@author: 03950025081
"""
import matplotlib.pyplot as plt 
from sol_analitica_estacionario import analitica
from sol_numerica_estacionario import estacionario
from scipy.interpolate import make_interp_spline
import numpy as np

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
    erro_10 = []
    erro_20 = []
    erro_30 = []
    sum = 0 
    
    for i in range(len(temperatura_analitica)):
        temp_analitico = temperatura_analitica[i]
        temp_estac = temperatura_estacionario[i]
        sum = 0
        if i == 0 :
            ui = temp_analitico[8]
            vi = temp_estac[1]
            nx = nx_list_estacionario[i]
            for j in range(1,len(ui)-1):
                uii = ui[j]
                vii = vi[j]
                sum = sum + abs((uii - vii) / uii) 
                erro_10.append(abs((uii - vii) / uii) *100)
            erro = 1/nx * sum 
            erro_percentual.append(erro)
        elif i == 1:
            ui = temp_analitico[18]
            vi = temp_estac[1]
            nx = nx_list_estacionario[i]
            for j in range(1,len(ui)-1):
                uii = ui[j]
                vii = vi[j]
                sum = sum + abs((uii - vii) / uii)
                erro_20.append(abs((uii - vii) / uii) *100)
            erro = 1/nx * sum 
            erro_percentual.append(erro)
        elif i == 2:
            ui = temp_analitico[28]
            vi = temp_estac[1]
            nx = nx_list_estacionario[i]
            for j in range(1,len(ui)-1):
                uii = ui[j]
                vii = vi[j]
                sum = sum + abs((uii - vii) / uii)
                erro_30.append(abs((uii - vii) / uii) *100)
            erro = 1/nx * sum 
            erro_percentual.append(erro)
    return temperatura_analitica, temperatura_estacionario, erro_percentual, nx_list_analitico, nx_list_estacionario, x_list_analitico, y_list_analitico, x_list_numerico, y_list_numerico, erro_10, erro_20, erro_30

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
    erro_10 = []
    erro_20 = []
    erro_30 = []
    sum = 0
    
    for i in range(len(temperatura_analitica)):
        temp_analitico = temperatura_analitica[i]
        temp_estac = temperatura_estacionario[i]
        sum = 0
        if i == 0 :
            ui = temp_analitico[:,7]
            vi = temp_estac[:,2]
            ny = ny_list_estacionario[i]
            vi_inv = vi[::-1]
            for j in range(1,len(ui)-1):
                uii = ui[j+1]
                vii = vi_inv[j]
                sum = sum + abs((uii - vii) / uii)
                erro_10.append(abs((uii - vii) / uii) *100)
            erro = 1/ny * sum 
            erro_percentual.append(erro)
        elif i == 1:
            ui = temp_analitico[:, 17]
            vi = temp_estac[:, 2]
            ny = ny_list_estacionario[i]
            vi_inv = vi[::-1]
            for j in range(1,len(ui)-1):
                uii = ui[j+1]
                vii = vi_inv[j]
                sum = sum + abs((uii - vii) / uii)
                erro_20.append(abs((uii - vii) / uii) *100)
            erro = 1/ny * sum 
            erro_percentual.append(erro)
        elif i == 2:
            ui = temp_analitico[:,27]
            vi = temp_estac[:,2]
            ny = ny_list_estacionario[i]
            vi_inv = vi[::-1]
            for j in range(1,len(ui)-1):
                uii = ui[j+1]
                vii = vi_inv[j]
                sum = sum + abs((uii - vii) / uii)
                erro_30.append(abs((uii - vii) / uii) *100)
            erro = 1/ny * sum 
            erro_percentual.append(erro)
    return temperatura_analitica, temperatura_estacionario, erro_percentual, ny_list_analitico, ny_list_estacionario, x_list_analitico, y_list_analitico, x_list_numerico, y_list_numerico, erro_10, erro_20, erro_30

temperatura_analitica_nx, tempratura_estacionario_nx, erro_variacao_nx, nx_list_analitico, nx_list_estacionario, x_list_analitico, y_list_analitico, x_list_numerico, y_list_numerico, erro_10_nx, erro_20_nx, erro_30_nx = calculate_variacao_nx()
temperatura_analitica_ny, tempratura_estacionario_ny, erro_variacao_ny, ny_list_analitico, ny_list_estacionario, x_list_analitico, y_list_analitico,  x_list_numerico, y_list_numerico, erro_10_ny, erro_20_ny, erro_30_ny = calculate_variacao_ny()

for i in range(len(temperatura_analitica_nx)):
    analitica2 = temperatura_analitica_nx[i]
    x_analitico = x_list_analitico[i]
    numerico = tempratura_estacionario_nx[i]
    x_numerico = x_list_numerico[i]
    xi = []
    ui =[]
    vi =[]
    if i == 0:
        x_a = x_analitico[4]
        y_a = analitica2[4]
        x_n = x_numerico[5]
        y_n = numerico[5]
        for j in range(len(x_a)):
            xi.append(x_a[j])
            ui.append(y_a[j])
            vi.append(y_n[j])
        spl_ui = make_interp_spline(xi, ui)
        spl_vi = make_interp_spline(xi, vi)
        xi_smooth = np.linspace(min(xi), max(xi), 300)
        ui_smooth = spl_ui(xi_smooth)
        vi_smooth = spl_vi(xi_smooth)
        plt.plot(xi_smooth, ui_smooth, label='SA - nx={}'.format(nx_list_analitico[i]))
        plt.plot(xi_smooth, vi_smooth, label='SE - nx={}'.format(nx_list_estacionario[i]))
    elif i == 1:
        x_a = x_analitico[12]
        y_a = analitica2[12]
        x_n = x_numerico[7]
        y_n = numerico[7]
        for j in range(len(x_a)):
            xi.append(x_a[j])
            ui.append(y_a[j])
            vi.append(y_n[j])
        spl_ui = make_interp_spline(xi, ui)
        spl_vi = make_interp_spline(xi, vi)
        xi_smooth = np.linspace(min(xi), max(xi), 300)
        ui_smooth = spl_ui(xi_smooth)
        vi_smooth = spl_vi(xi_smooth)
        plt.plot(xi_smooth, ui_smooth, label='SA - nx={}'.format(nx_list_analitico[i]))
        plt.plot(xi_smooth, vi_smooth, label='SE - nx={}'.format(nx_list_estacionario[i]))
    elif i == 2:
        x_a = x_analitico[21]
        y_a = analitica2[21]
        x_n = x_numerico[8]
        y_n = numerico[8]
        for j in range(len(x_a)):
            xi.append(x_a[j])
            ui.append(y_a[j])
            vi.append(y_n[j])
        spl_ui = make_interp_spline(xi, ui)
        spl_vi = make_interp_spline(xi, vi)
        xi_smooth = np.linspace(min(xi), max(xi), 300)
        ui_smooth = spl_ui(xi_smooth)
        vi_smooth = spl_vi(xi_smooth)
        plt.plot(xi_smooth, ui_smooth, label='SA - nx={}'.format(nx_list_analitico[i]))
        plt.plot(xi_smooth, vi_smooth, label='SE - nx={}'.format(nx_list_estacionario[i]))
plt.legend(loc='upper right', fontsize='small')
plt.xlabel('Comprimento [m]')
plt.ylabel('Temperatura [°C]')



for i in range(len(temperatura_analitica_ny)):
    analitica2 = temperatura_analitica_ny[i]
    x_analitico = x_list_analitico[i]
    numerico = tempratura_estacionario_ny[i]
    x_numerico = x_list_numerico[i]
    xi = []
    ui =[]
    vi =[]
    if i == 0:
        x_a = x_analitico[1]
        y_a = analitica2[:, 1]
        x_n = x_numerico[8]
        y_n = numerico[:, 8]
        for j in range(len(x_a)):
            xi.append(x_a[j])
            ui.append(y_a[j])
            vi.append(y_n[j])
        vi_inv = vi[::-1]
        spl_ui = make_interp_spline(xi, ui)
        spl_vi = make_interp_spline(xi, vi_inv)
        xi_smooth = np.linspace(min(xi), max(xi), 300)
        ui_smooth = spl_ui(xi_smooth)
        vi_smooth = spl_vi(xi_smooth)
        plt.plot(xi_smooth, ui_smooth, label='SA - ny={}'.format(nx_list_analitico[i]))
        plt.plot(xi_smooth, vi_smooth, label='SE - ny={}'.format(nx_list_estacionario[i]))
    elif i == 1:
        x_a = x_analitico[4]
        y_a = analitica2[:, 4]
        x_n = x_numerico[15]
        y_n = numerico[:, 15]
        for j in range(len(x_a)):
            xi.append(x_a[j])
            ui.append(y_a[j])
            vi.append(y_n[j])
        vi_inv = vi[::-1]
        spl_ui = make_interp_spline(xi, ui)
        spl_vi = make_interp_spline(xi, vi_inv)
        xi_smooth = np.linspace(min(xi), max(xi), 300)
        ui_smooth = spl_ui(xi_smooth)
        vi_smooth = spl_vi(xi_smooth)
        plt.plot(xi_smooth, ui_smooth, label='SA - ny={}'.format(nx_list_analitico[i]))
        plt.plot(xi_smooth, vi_smooth, label='SE - ny={}'.format(nx_list_estacionario[i]))
    elif i == 2:
        x_a = x_analitico[14]
        y_a = analitica2[:, 14]
        x_n = x_numerico[15]
        y_n = numerico[:, 15]
        for j in range(len(x_a)):
            xi.append(x_a[j])
            ui.append(y_a[j])
            vi.append(y_n[j])
        vi_inv = vi[::-1]
        spl_ui = make_interp_spline(xi, ui)
        spl_vi = make_interp_spline(xi, vi_inv)
        xi_smooth = np.linspace(min(xi), max(xi), 300)
        ui_smooth = spl_ui(xi_smooth)
        vi_smooth = spl_vi(xi_smooth)
        plt.plot(xi_smooth, ui_smooth, label='SA - ny={}'.format(ny_list_analitico[i]))
        plt.plot(xi_smooth, vi_smooth, label='SE - ny={}'.format(ny_list_estacionario[i]))

plt.legend(loc='upper left', fontsize='small')
plt.xlabel('Comprimento [m]')
plt.ylabel('Temperatura [°C]')
plt.show()


for i in range(len(x_list_analitico)):
    x_plot = x_list_analitico[i]
    x = x_plot[5]
    if i == 0 :
        x_10 = []
        for j in range(1,len(x)-1):
            x_10.append(x[j])
        plt.plot(x_10, erro_10_nx, label='Erro nx={}'.format(nx_list_analitico[i]))

    elif i == 1 :
        x_20 = []
        for j in range(1,len(x)-1):
            x_20.append(x[j])
        plt.plot(x_20, erro_20_nx, label='Erro nx={}'.format(nx_list_analitico[i]))
    elif i == 2 :
        x_30 = []
        for j in range(1,len(x)-1):
            x_30.append(x[j])
        plt.plot(x_30, erro_30_nx, label='Erro nx={}'.format(nx_list_analitico[i]))

plt.legend()
plt.xlabel('Comprimento [m]')
plt.ylabel('Erro Percentual Verdadeiro Local')
plt.show()
 


for i in range(len(x_list_analitico)):
    x_plot = y_list_analitico[i]
    x = x_plot[:, 5]
    if i == 0 :
        x_10 = []
        for j in range(1,len(x)-1):
            x_10.append(x[j])
        plt.plot(x_10, erro_10_ny, label='Erro ny={}'.format(ny_list_analitico[i]))
    elif i == 1 :
        x_20 = []
        for j in range(1,len(x)-1):
            x_20.append(x[j])
        plt.plot(x_20, erro_20_ny, label='Erro ny={}'.format(ny_list_analitico[i]))
    elif i == 2 :
        x_30 = []
        for j in range(1,len(x)-1):
            x_30.append(x[j])
        plt.plot(x_30, erro_30_ny, label='Erro ny={}'.format(ny_list_analitico[i]))

plt.legend()
plt.xlabel('Comprimento [m]')
plt.ylabel('Erro Percentual Verdadeiro Local')
plt.show()
