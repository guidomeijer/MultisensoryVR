import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths

path_dict = paths()

waveform_si = pd.read_csv(path_dict['save_path'] / 'waveform_metrics_si.csv')
waveform_calc = pd.read_csv(path_dict['save_path'] / 'waveform_metrics_calc.csv')

waveform_si = waveform_si[waveform_si['good'] == 1]
waveform_calc = waveform_calc[waveform_calc['good'] == 1]

f, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
ax1.scatter(waveform_si['peak_to_valley'] * 1000, waveform_calc['spike_width'], s=3)
ax1.set(xlim=[0, 1.5], ylim=[0, 1.5], xlabel='Spike interface', ylabel='Own calculation',
        title='Spike width')
plt.tight_layout()
