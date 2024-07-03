# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:42:24 2024

@author: 03950025081
"""

from term_font import term_font 

# Xisto
rho = 2.7 # g/cm³◘
cp = 0.19 # cal/(g.°C)
k = 0.0007 # cal/(cm.s.°C)
material = 'Xisto'

nx = 20
T_xis, temp_xis = term_font.calculate_gs(nx, rho, cp, k, material)

# basalto 
rho = 2.9 # kg/m³
cp = 0.2 # cal/(g.°C)
k = 0.004 # cal/(cm.s.°C)
material = 'Basalto'

nx = 20
T_bas, temp_bas = term_font.calculate_gs(nx, rho, cp, k, material)

# Arenito  
rho = 2.65 # g/cm³◘
cp = 0.22  # cal/(g.°C)
k = 0.002  # cal/(cm.s.°C)
material = 'Arenito'

nx = 20
T_are, temp_are = term_font.calculate_gs(nx, rho, cp, k, material)

