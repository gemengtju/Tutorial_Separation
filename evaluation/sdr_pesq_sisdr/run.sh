#! /bin/bash

# Copyright 2017
# Author: Chenglin Xu (NTU, Singapore)
# Email: xuchenglin28@gmail.com
# Updated by Chenglin, Dec 2018

/export/home/clx214/Matlab_R2014A/bin/matlab -nodesktop -nosplash -r "eval_sdr('tt', 0, 0, 'Ext_mfcc_Mix_N256_L20_1L80_2L160_S10_B256_H512_P3_X8_R4_C2_gln_si-sdr_sigmoid_deconv_BLSTM_e400_spk0.2_mscmo_a0.1_b0.1', 'mix', 's1')"
