import numpy as np

from erros_metricas_L2 import erros_tt_ftcs
from erros_metricas_L2 import erros_tt_btcs
from erros_metricas_L2 import erros_tt_cn
from questao_2 import tempo_computacional_tf

calc_questao_1_ftcs_tempo = erros_tt_ftcs.calculate_erros_tempo()
calc_questao_1_btcs_tempo = erros_tt_btcs.calculate_erros_tempo()
calc_questao_1_cn_tempo = erros_tt_cn.calculate_erros_tempo()
calc_questao_1_ftcs_malha = erros_tt_ftcs.calculate_erros_malha()
calc_questao_1_btcs_malha = erros_tt_btcs.calculate_erros_malha()
calc_questao_1_cn_malha = erros_tt_cn.calculate_erros_malha()

calc_questao_2_tempo_comp_ht = tempo_computacional_tf.calculate_tempo_computacional_h_t()
calc_questao_2_tempo_comp_hx = tempo_computacional_tf.calculate_tempo_computacional_h_x()