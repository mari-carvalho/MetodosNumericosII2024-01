# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:26:13 2024

@author: 03950025081
"""

from term_font import term_font

temp_comp_gs = []
temp_comp_jac = []
temp_comp_gsr = []


# Arenito  
rho = 2.65 # g/cm³◘
cp = 0.22  # cal/(g.°C)
k = 0.002  # cal/(cm.s.°C)
material = 'Arenito'

nx = 5 
T_new_gs, temp_simu_gs = term_font.calculate_gs(nx, rho, cp, k, material)
temp_comp_gs.append(temp_simu_gs)


'''
nx = 20
T_new_gs, temp_simu_gs = term_font.calculate_gs(nx, rho, cp, k, material)
temp_comp_gs.append(temp_simu_gs)
'''
'''
nx = 40 
T_new_gs, temp_simu_gs = term_font.calculate_gs(nx, rho, cp, k, material)
temp_comp_gs.append(temp_simu_gs)
'''

nx = 60 
T_new_gs, temp_simu_gs = term_font.calculate_gs(nx, rho, cp, k, material)
temp_comp_gs.append(temp_simu_gs)

'''
nx = 5
T_new_jac, temp_simu_jac = term_font.calculate_jac(nx, rho, cp, k, material)
temp_comp_jac.append(temp_simu_jac)
nx = 20
T_new_jac, temp_simu_jac = term_font.calculate_jac(nx, rho, cp, k, material)
temp_comp_jac.append(temp_simu_jac)
nx = 40
T_new_jac, temp_simu_jac = term_font.calculate_jac(nx, rho, cp, k, material)
temp_comp_jac.append(temp_simu_jac)
nx = 60
T_new_jac, temp_simu_jac = term_font.calculate_jac(nx, rho, cp, k, material)
temp_comp_jac.append(temp_simu_jac)

nx = 5
T_new_gsr, temp_simu_gsr = term_font.calculate_gsr(nx, rho, cp, k, material)
temp_comp_gsr.append(temp_simu_gsr)
nx = 20
T_new_gsr, temp_simu_gsr = term_font.calculate_gsr(nx, rho, cp, k, material)
temp_comp_gsr.append(temp_simu_gsr)
nx = 40
T_new_gsr, temp_simu_gsr = term_font.calculate_gsr(nx, rho, cp, k, material)
temp_comp_gsr.append(temp_simu_gsr)
nx = 60
T_new_gsr, temp_simu_gsr = term_font.calculate_gsr(nx, rho, cp, k, material)
temp_comp_gsr.append(temp_simu_gsr)
'''