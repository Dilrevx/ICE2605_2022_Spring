from re import A
from turtle import title
from wave import Wave_write
import scipy.signal
from typing import Iterable, List
import scipy.io.wavfile
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.fft

PFIG = "figure/"
PLT_RATE = 48000*4  # higher rate for plot or sth

sr, xn = scipy.io.wavfile.read("1.wav")
xn: np.ndarray
duration = xn.shape[0]/sr

# Interpolation
xt_sample = scipy.interpolate.interp1d(np.arange(xn.shape[0]), xn, 'zero',
                                       bounds_error=False, fill_value=(xn[0], xn[-1]))


def xt(t): return xt_sample(t*sr)  # define a function of t


t4 = np.linspace(0, duration, int(duration*PLT_RATE)
                 )  # generate time seq for plot
t = np.linspace(0, duration, xn.shape[0])

plt.plot(t4, xt(t4))
plt.savefig(os.path.join(PFIG, "1_interpolation.pdf"))


# Affine Composition
def Rxt(t): return xt(-t)
def x2t(t): return xt(2*t)
def xt2(t): return xt(t/2)


# Plot Affine
# plot 4 graph piling vertically
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
ax: List[plt.Axes]

ax[0].plot(t4, xt(t4))
ax[1].plot(-t4, Rxt(-t4))
ax[2].plot(t4/2, x2t(t4/2))  # scale time axis to fit
ax[3].plot(2*t4, xt2(2*t4))

ax[0].set(title='x(t)')
ax[1].set(title='x(-t)')
ax[2].set(title='x(2t)')
ax[3].set(title='x(t/2)')

fig.tight_layout()
plt.savefig(os.path.join(PFIG, "2_scale.pdf"))


# Fourier 3
# Solution: FFT
Xd = scipy.fft.fft(xn)  # Xd([0:2Pi])
Xd: np.ndarray

Rxn = xn[::-1]  # sample to generate discrete seq for fft, X is the same
x2n = xn[::2]
xn2 = xt2(np.linspace(0, 2*duration, xn.shape[0]*2))

RX = scipy.fft.fft(Rxn)
X2n = scipy.fft.fft(x2n)
Xn2 = scipy.fft.fft(xn2)


def plot_mag_phase(X, s):
    fig, ax = plt.subplots(2, 1)
    w = np.linspace(0, 2*np.pi, X.shape[0])
    ax[0].plot(w, np.abs(X))
    ax[0].set(title='magnitude')

    ax[1].plot(w, np.angle(X))
    ax[1].set(title='phase')

    fig.tight_layout()
    plt.savefig(os.path.join(PFIG, s))


plot_mag_phase(Xd, '3_xt.pdf')
plot_mag_phase(RX, '3_Rx.pdf')
plot_mag_phase(X2n, '3_x2t.pdf')
plot_mag_phase(Xn2, '3_xt2.pdf')




# Fourier 4
Xd_mag = np.abs(Xd)  # preserve mag only
Xd_arg = Xd / Xd_mag  # arg only

x_mag = scipy.fft.ifft(Xd_mag)  # inverse fft
x_arg = scipy.fft.ifft(Xd_arg)
x_mag: np.ndarray
x_arg: np.ndarray

x_mag = np.real(x_mag)  # real signal
x_arg = np.real(x_arg)


'float output is horrible !!!'
MAG_FACTOR = (2**16 - 1)/max(x_mag) * 1  # 16 is a better parameter
ARG_FACTOR = (2**16-1)/max(x_arg) * 1  # 4

x_mag = np.round(x_mag*MAG_FACTOR)  # Scale to int16
x_mag = x_mag.astype('int16')
x_arg = np.round(x_arg*ARG_FACTOR)
x_arg = x_arg.astype('int16')

# Write audio
scipy.io.wavfile.write('1_mag.wav', sr, x_mag)
scipy.io.wavfile.write('1_arg.wav', sr, x_arg)
# scipy.io.wavfile.write('test.wav',sr,np.round(scipy.fft.ifft(Xd)).astype("int16"))

# Plot waveform
fig, ax = plt.subplots(3, 1, sharex=True)  # Create sub fig

ax[0].plot(t, xn)  # respectly plot 3 signals in the subfig
ax[1].plot(t, x_mag)  # Better if scale x 16
ax[2].plot(t, x_arg)  # Better scale x 4

ax[0].set(title='waveform Mag & Arg')
ax[1].set(title='wavefrom of only Mag')
ax[2].set(title='waveform of only Arg')

fig.tight_layout()
plt.savefig(os.path.join(PFIG, "4_Mag&Fig.pdf"))


# Low-pass Filter
CUT_FREQ = np.pi/2  # cut off freq, [0,2pi]


def H_ideal(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    cut_left = int(CUT_FREQ/2/np.pi * x.shape[0])
    cut_right = int(len(x)-CUT_FREQ/2/np.pi * x.shape[0])
    for i in range(cut_left, cut_right):
        x[i] = 0        # Elimate those out-band points
    return x


Xd_filtered = H_ideal(Xd)

x_filtered = scipy.fft.ifft(Xd_filtered)  # inverse fft for plot
x_filtered = np.real(x_filtered)

fig, ax = plt.subplots(3, 1)

ax[0].plot(t, x_filtered)  # plot waveform
ax[1].plot(np.linspace(0, 2*np.pi, Xd_filtered.shape[0]),
           np.abs(Xd_filtered))  # plot spectrum, mag, angle respectly
ax[2].plot(np.linspace(0, 2*np.pi, Xd_filtered.shape[0]),
           np.angle(Xd_filtered))

ax[0].set(title='waveform')
ax[1].set(title='spectrum-magnitude')
ax[2].set(title='spectrum-phase')

fig.tight_layout()
plt.savefig(os.path.join(PFIG, '5_filter.pdf'))
plt.show()
