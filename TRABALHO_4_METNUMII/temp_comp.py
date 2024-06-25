# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:26:13 2024

@author: 03950025081
"""

from term_font import temp_comp

temp_comp_gs = []
temp_comp_jac = []
temp_comp_gsr = []

# titanio
rho = 4.5 # g/cm³◘
cp = 0.1247 # cal/(g.°C)
k = 0.0523 # cal/(cm.s.°C)

# tungstênio 
rho = 19.3 # kg/m³
cp = 0.0315 # cal/(g.°C)
k = 0.4156 # cal/(cm.s.°C)

# cobre  
rho = 8.92  # g/cm³◘
cp = 0.092  # cal/(g.°C)
k = 0.95  # cal/(cm.s.°C)

nx = 5 
T_new_gs, temp_simu_gs = temp_comp.calculate_gs(nx)
temp_comp_gs.append(temp_simu_gs)
nx = 20
T_new_gs, temp_simu_gs = temp_comp.calculate_gs(nx)
temp_comp_gs.append(temp_simu_gs)
nx = 40 
T_new_gs, temp_simu_gs = temp_comp.calculate_gs(nx)
temp_comp_gs.append(temp_simu_gs)
nx = 60 
T_new_gs, temp_simu_gs = temp_comp.calculate_gs(nx)
temp_comp_gs.append(temp_simu_gs)


nx = 5
T_new_jac, temp_simu_jac = temp_comp.calculate_jac(nx)
temp_comp_jac.append(temp_simu_jac)
nx = 20
T_new_jac, temp_simu_jac = temp_comp.calculate_jac(nx)
temp_comp_jac.append(temp_simu_jac)
nx = 40
T_new_jac, temp_simu_jac = temp_comp.calculate_jac(nx)
temp_comp_jac.append(temp_simu_jac)
nx = 60
T_new_jac, temp_simu_jac = temp_comp.calculate_jac(nx)
temp_comp_jac.append(temp_simu_jac)


nx = 5
T_new_gsr, temp_simu_gsr = temp_comp.calculate_gsr(nx)
temp_comp_gsr.append(temp_simu_gsr)
nx = 20
T_new_gsr, temp_simu_gsr = temp_comp.calculate_gsr(nx)
temp_comp_gsr.append(temp_simu_gsr)
nx = 40
T_new_gsr, temp_simu_gsr = temp_comp.calculate_gsr(nx)
temp_comp_gsr.append(temp_simu_gsr)
nx = 60
T_new_gsr, temp_simu_gsr = temp_comp.calculate_gsr(nx)
temp_comp_gsr.append(temp_simu_gsr)