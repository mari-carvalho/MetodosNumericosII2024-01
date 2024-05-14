import numpy as np
import matplotlib.pyplot as plt 
import sympy as sp 
from CN_tt import CN
from solver_gauss_seidel import gauss_seidel
from BTCS import BTCS

# Propreidades do Material - Cobre:
rho = 8.92  # g/cm³
cp = 0.092  # cal/(g.ºC)
k = 0.95  # cal/(cm.s.ºC)
L = 80  # cm
T0 = 50  # ºC
Tw = 0  # ºC
Te = 0  # ºC
t0 = 0
tf = 100
x0 = 0
xf = L
qw = 25
h_t = 2
h_x = 4
n_t = ((tf-t0)/(h_t))
n_x = ((xf-x0)/(h_x))

calc_BTCS = BTCS.calculate_BTCS_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, h_t, h_x, n_t, n_x)
calc_CN= CN.calculate_CN_tt(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, h_t, h_x, n_t, n_x)
calc_CN= CN.calculate_CN_tf(rho, cp, k, L, Tw, T0, Te, x0, xf, t0, tf, qw, h_t, h_x, n_t, n_x)
