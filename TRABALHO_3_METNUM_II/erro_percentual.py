# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:45:18 2024

@author: 03950025081
"""

from sol_analitica_estacionario import analitica
from sol_numerica_estacionario import estacionario

# Variando NX:
    
temperatura_analitica = []
nx_list_analitico = []
temperatura_estacionario = []
nx_list_estacionario = []

# Analítica

analitica_nx10, nx = analitica.calculate_analitica_nx10()
temperatura_analitica.append(analitica_nx10)
nx_list_analitico.append(nx)

analitica_nx20, nx = analitica.calculate_analitica_nx20()
temperatura_analitica.append(analitica_nx20)
nx_list_analitico.append(nx)

analitica_nx30, nx = analitica.calculate_analitica_nx30()
temperatura_analitica.append(analitica_nx30)
nx_list_analitico.append(nx)

# Numérico

estacionario_nx10, nx = estacionario.calculate_estacionario_nx10()
temperatura_estacionario.append(estacionario_nx10)
nx_list_estacionario.append(nx)

estacionario_nx20, nx = estacionario.calculate_estacionario_nx20()
temperatura_estacionario.append(estacionario_nx20)
nx_list_estacionario.append(nx)

estacionario_nx30, nx = estacionario.calculate_estacionario_nx30()
temperatura_estacionario.append(estacionario_nx30)
nx_list_estacionario.append(nx)

# Cálculo do Erro 

erro_percentual = []
sum = 0 

for i in range(len(temperatura_analitica)):
    temp_analitico = temperatura_analitica[i]
    temp_estac = temperatura_estacionario[i]
    if i == 0 :
        ui = temp_analitico[8]
        vi = temp_estac[0]
    elif i == 1:
        ui = temp_analitico[18]
        vi = temp_estac[0]
    elif i == 2:
        ui = temp_analitico[28]
        vi = temp_estac[0]
    nx = nx_list_estacionario[i]
    for j in range(1,len(ui)-1):
        uii = ui[j]
        vii = vi[j]
        sum = sum + abs((ui[j] - vi[j]) / ui[j])
    erro = 1/nx * sum * 100
    erro_percentual.append(erro)

'''    
# Variando NY:
    
temperatura_analitica = []
nx_list_analitico = []
temperatura_estacionario = []
nx_list_estacionario = []

# Analítica

analitica_nx10, nx = analitica.calculate_analitica_nx10()
temperatura_analitica.append(analitica_nx10)
nx_list_analitico.append(nx)

analitica_nx20, nx = analitica.calculate_analitica_nx20()
temperatura_analitica.append(analitica_nx20)
nx_list_analitico.append(nx)

analitica_nx30, nx = analitica.calculate_analitica_nx30()
temperatura_analitica.append(analitica_nx30)
nx_list_analitico.append(nx)

# Numérico

estacionario_nx10, nx = estacionario.calculate_estacionario_nx10()
temperatura_estacionario.append(estacionario_nx10)
nx_list_estacionario.append(nx)

estacionario_nx20, nx = estacionario.calculate_estacionario_nx20()
temperatura_estacionario.append(estacionario_nx20)
nx_list_estacionario.append(nx)

estacionario_nx30, nx = estacionario.calculate_estacionario_nx30()
temperatura_estacionario.append(estacionario_nx30)
nx_list_estacionario.append(nx)

# Cálculo do Erro 

erro_percentual = []

for i in range(len(temperatura_analitica)):
    temp_analitico = temperatura_analitica[i]
    temp_estac = temperatura_estacionario[i]
    ui = temp_analitico[7]
    vi = temp_estac[7]
    nx = nx_list_estacionario[i]
    for j in range(len(ui)):
        sum = abs((ui[j] - vi[j]) / ui[j])
    erro = 1/nx * sum 
    erro_percentual.append(erro)
    '''