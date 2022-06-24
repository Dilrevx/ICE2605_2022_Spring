import os
import scipy.io.wavfile
import scipy.interpolate
import numpy as np
from matplotlib import pyplot as plt
PFIG = 'figure/'

sr, xn = scipy.io.wavfile.read('2.wav')
xn: np.ndarray
duration = xn.shape[0]/sr

# interpolate and sample at 44kHz again
x_sample = scipy.interpolate.interp1d(np.arange(xn.shape[0]), xn, 'linear',
                                      bounds_error=False, fill_value=(xn[0], xn[-1]))

sr44 = int(44e3)
xn44 = x_sample(np.linspace(0, int(duration*sr), int(duration *
                sr44), dtype='float64'))  # xn44: sample at 44kHz


# sr44 = sr
# xn44 = xn  # start at 44.1

# define a array, we are appending 22, 11, 5.5, 2.75
sample_arr = [(xn44, sr44)]
last_sr = sr44
scipy.io.wavfile.write('2_' + str(last_sr/1e3) + 'kHz.wav', last_sr, xn44)

while True:  # iteratively append 22,11,5.5,2.75
    last_sr //= 2
    if last_sr < 2.75e3 : break

    sample_arr.append((sample_arr[-1][0][::2], last_sr))
    # output
    scipy.io.wavfile.write('2_' + str(last_sr/1e3) +
                           'kHz.wav', last_sr, sample_arr[-1][0])


# Reconstruct
for xn_i, sr_i in sample_arr:
    # interpolate to increase sample rate
    x_sample = scipy.interpolate.interp1d(np.arange(xn_i.shape[0]), xn_i, 'linear',
                                          bounds_error=False, fill_value=(xn_i[0], xn_i[-1]))
    signal_construct = x_sample(
        np.linspace(0, round(duration * sr_i), round(duration * sr), dtype='float64'))

    np.round(signal_construct)
    signal_construct.astype('int16') # convert to int, float is horrible

    scipy.io.wavfile.write('reconstruct_'+str(sr_i/1e3) +
                           'kHz.wav', sr, signal_construct)

    t = np.linspace(0, duration, signal_construct.shape[0])

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, signal_construct)
    plt.savefig(os.path.join(PFIG, '2_'+str(sr_i/1e3)+'kHz.pdf'))

    # compute error
    error = np.sum(signal_construct-xn)/np.sum(xn)
    print(str(sr_i/1e3),'\t',error)