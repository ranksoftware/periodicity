import numpy as np
import matplotlib.pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from scipy.stats import gamma
import matplotlib.mlab as mlab

# parameters
sample_rate = 1024
num_seconds = 1
num_samples = (num_seconds - 0) * sample_rate

# x-axis time values
time = np.linspace(0, num_seconds, num_samples)

# main waveform 1
freq1 = 60
magnitude1 = 110
waveform1 = magnitude1 * np.sin(2 * pi * freq1 * time)

# main waveform 2
freq2 = 40
magnitude2 = 80
waveform2 = magnitude2 * np.sin(2 * pi * freq2 * time)

# main waveform 3
freq3 = 100
magnitude3 = 150
waveform3 = magnitude3 * np.sin(2 * pi * freq3 * time)

# main waveform 4
freq4 = 130
magnitude4 = 10
waveform4 = magnitude4 * np.sin(2 * pi * freq4 * time)

# additive noise
noise_mag = 3
noise = np.random.normal(0, noise_mag, num_samples)

# combine together
time_data = waveform1 + waveform2 + waveform3 + waveform4 + noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefig("C:/Users/hazem/Dropbox/RANK-Dev/ML/periodicity_Talk_Material/plots/periodic_signal_multiple.png")
plt.show()

# frequency domain
frequency = np.linspace(0.0, sample_rate / 2, int(num_samples / 2))
freq_data = fft(time_data)
freq_mag = 2 / num_samples * np.abs(freq_data[0:np.int(num_samples / 2)])

# plot frequency domain
plt.figure(figsize=(17, 8))
plt.plot(frequency, freq_mag)
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.savefig("C:/Users/hazem/Dropbox/RANK-Dev/ML/periodicity_Talk_Material/plots/periodic_Spectrum_multiple.png")
plt.show()

# plot histogram
plt.figure(figsize=(17, 8))

# best fit of data
# (mu, sigma) = norm.fit(freq_mag)
fit_alpha, fit_loc, scale = gamma.fit(freq_mag)
# the histogram of the data
n, bins, patches = plt.hist(freq_mag, bins=30, density=True, facecolor='green', alpha=0.75)
# add a 'best fit' line
# y = mlab.normpdf(bins, mu, sigma)
y = gamma.pdf(bins, a=fit_alpha, loc=fit_loc, scale=scale)
plt.plot(bins, y, 'r--', linewidth=2)

plt.title('Frequency domain Signal Histogram')
plt.xlabel('Frequency in Hz')
plt.ylabel('Probability')
plt.ylim([0, 1])
plt.show()
