# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:45:18 2024

@author: 03950025081
"""

from sol_analitica_estacionario import analitica
from sol_numerica_estacionario import estacionario

# Variando NX:

analitica_nx10 = analitica.calculate_analitica_nx10()
analitica_nx20 = analitica.calculate_analitica_nx20()
analitica_nx30 = analitica.calculate_analitica_nx30()
estacionario_nx10 = estacionario.calculate_estacionario_nx10()
estacionario_nx20 = estacionario.calculate_estacionario_nx20()
estacionario_nx30 = estacionario.calculate_estacionario_nx30()

