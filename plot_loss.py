import numpy as np
import matplotlib.pyplot as plt

SEA_a = np.load('/home/wbo/Project/SEA/train/save_m/SEA_a128_train_history.npz')
SEA_a_loss = SEA_a['val_loss']
SEA_b = np.load('/home/wbo/Project/SEA/train/save_m/SEA_b128_train_history.npz')
SEA_b_loss = SEA_b['val_loss']
SEA_c = np.load('/home/wbo/Project/SEA/train/save_m/SEA_c128_train_history.npz')
SEA_c_loss = SEA_c['val_loss']
SECA_c = np.load('/home/wbo/Project/SEA/train/save_m/SECA_c_002_history.npz')
SECA_c_loss = SECA_c['val_loss']
CVLAD = np.load('/home/wbo/Project/SEA/train/save_m/CenterVLAD_train_history.npz')
CVLAD_loss = CVLAD['val_loss']
ATR = np.load('/home/wbo/Project/SEA/train/save_m/SEA_train_history.npz')
ATR_loss = ATR['val_loss']

plt.figure(figsize=(15,10))

plt.xlabel('epoch', fontsize=16)
plt.ylabel('loss', fontsize=16)

plt.plot(SEA_a_loss, label='SEA_a')
plt.plot(SEA_b_loss, label='SEA_b')
plt.plot(SEA_c_loss, label='SEA_c')
plt.plot(SECA_c_loss, label='SECA_c')
plt.plot(ATR_loss, label='SEA')
# plt.plot(CVLAD_loss, label='CVLAD_loss')
# plt.plot(xx, Cvlad, marker='*', label='CenterVLAD', markersize=8)
# plt.plot(xx, SeC_c, marker='^', label='SECA_c', markersize=8)
# plt.plot(xx, Se_c, marker='d',label='SEA_c', markersize=8)
plt.xlim(0, 50)
plt.legend(loc='best', fontsize='x-large')

# plt.savefig('./p_loss')
plt.show()
