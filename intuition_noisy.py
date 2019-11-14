import numpy as np
import matplotlib.pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from scipy.stats import gamma
import matplotlib.mlab as mlab

# parameters
sample_rate = 1024
num_seconds = 3
num_samples = (num_seconds - 0) * sample_rate

# x-axis time values
time = np.linspace(0, num_seconds, num_samples)

# initialize time domain signal
time_data = np.zeros((num_samples))

num_noise_sources = 1000
noise_mag = 110 / num_noise_sources
# additive noise
for i in range(num_noise_sources):
    noise = np.random.normal(0, noise_mag, num_samples)
    # combine together
    time_data = time_data + noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
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
plt.xlabel('Fourier Amplitude')
plt.ylabel('Probability')
plt.show()
