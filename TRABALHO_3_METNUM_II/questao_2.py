# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 15:22:40 2024

@author: 03950025081
"""

from sol_numerica_transiente import transiente

temp_simu_gs = []
temp_simu_jac = []
temp_simu_gsr = []

temp_simu_gs_10 = transiente.calculate_transiente_nx10_gs()
temp_simu_gs.append(temp_simu_gs_10)
'''
temp_simu_gs_20 = transiente.calculate_transiente_nx20_gs()
temp_simu_gs.append(temp_simu_gs_20)
temp_simu_gs_30 = transiente.calculate_transiente_nx30_gs()
temp_simu_gs.append(temp_simu_gs_30)
temp_simu_jac_10 = transiente.calculate_transiente_nx10_jac()
temp_simu_jac.append(temp_simu_jac_10)
temp_simu_jac_20  = transiente.calculate_transiente_nx20_jac()
temp_simu_jac.append(temp_simu_jac_20)
temp_simu_jac_30  = transiente.calculate_transiente_nx30_jac()
temp_simu_jac.append(temp_simu_jac_30)
temp_simu_gsr_10  = transiente.calculate_transiente_nx10_gsr()
temp_simu_gsr.append(temp_simu_gsr_10)
temp_simu_gsr_20 = transiente.calculate_transiente_nx20_gsr()
temp_simu_gsr.append(temp_simu_gsr_20)
temp_simu_gsr_30  = transiente.calculate_transiente_nx30_gsr()
temp_simu_gsr.append(temp_simu_gsr_30)
'''
'''
temp_simu_gs_nt = []
temp_simu_jac_nt = []
temp_simu_gsr_nt = []

temp_simu_gs_10 = transiente.calculate_transiente_nt10_gs()
temp_simu_gs_nt.append(temp_simu_gs_10)
temp_simu_gs_50 = transiente.calculate_transiente_nt50_gs()
temp_simu_gs_nt.append(temp_simu_gs_50)
temp_simu_gs_100 = transiente.calculate_transiente_nt100_gs()
temp_simu_gs_nt.append(temp_simu_gs_100)
temp_simu_jac_10 = transiente.calculate_transiente_nt10_jac()
temp_simu_jac_nt.append(temp_simu_jac_10)
temp_simu_jac_50 = transiente.calculate_transiente_nt50_jac()
temp_simu_jac_nt.append(temp_simu_jac_50)
temp_simu_jac_100 = transiente.calculate_transiente_nt100_jac()
temp_simu_jac_nt.append(temp_simu_jac_100)
temp_simu_gsr_10 = transiente.calculate_transiente_nt10_gsr()
temp_simu_gsr_nt.append(temp_simu_gsr_10)
temp_simu_gsr_50 = transiente.calculate_transiente_nt50_gsr()
temp_simu_gsr_nt.append(temp_simu_gsr_50)
temp_simu_gsr_100 = transiente.calculate_transiente_nt100_gsr()
temp_simu_gsr_nt.append(temp_simu_gsr_100)
'''