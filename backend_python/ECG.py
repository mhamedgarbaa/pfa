### Import required packages
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Helps to obtain the FFT
import scipy.fftpack    
# Various operations on signals (waveforms)
import scipy.signal as signal

### Obtain ecg sample from csv file using pandas ###
dataset = pd.read_csv("noise.csv")
y = [e for e in dataset.hart]

# Number of samplepoints
N = len(y)
# sample spacing
Fs = 1000
T = 1.0 / Fs
# Compute x-axis
x = np.linspace(0.0, N*T, N)

# Compute FFT
yf = scipy.fftpack.fft(y)
# Compute frequency x-axis
xf = np.linspace(0.0, 1.0/(2.0*T), int(N//2))  # Fixed integer division

### Declare plots for time-domain and frequency-domain plots ###
fig_td = plt.figure(num='Time domain signals')
fig_fd = plt.figure(num='Frequency domain signals')

ax1 = fig_td.add_subplot(211)
ax1.set_title('Before filtering')
ax2 = fig_td.add_subplot(212)
ax2.set_title('After filtering')
ax3 = fig_fd.add_subplot(211)
ax3.set_title('Before filtering')
ax4 = fig_fd.add_subplot(212)
ax4.set_title('After filtering')

# Plot non-filtered inputs
ax1.plot(x, y, color='r', linewidth=0.7)
ax3.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='r', linewidth=0.7, label='raw')
ax3.set_ylim([0, 0.2])

### Compute filtering coefficients ###
b, a = signal.butter(4, 50/(Fs/2), 'low')

# Compute filtered signal
tempf = signal.filtfilt(b, a, y)
yff = scipy.fftpack.fft(tempf)

### Compute Kaiser window coefficients ###
nyq_rate = Fs / 2.0
width = 5.0/nyq_rate
ripple_db = 60.0
O, beta = signal.kaiserord(ripple_db, width)
cutoff_hz = 4.0

# Create lowpass FIR filter
taps = signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
y_filt = signal.lfilter(taps, 1.0, tempf)
yff = scipy.fftpack.fft(y_filt)

# Plot filtered outputs
ax4.plot(xf, 2.0/N * np.abs(yff[:N//2]), color='g', linewidth=0.7)
ax4.set_ylim([0, 0.2])
ax2.plot(x, y_filt, color='g', linewidth=0.7)

### Compute beats ###
dataset['filt'] = y_filt

# Calculate moving average
hrw = 1  # One-sided window size
fs = Fs  # Corrected sampling frequency (was 333)

window_size = int(hrw * fs)
# Ensure window size is at least 1 and not larger than data length
window_size = max(1, min(window_size, len(dataset.filt)))

mov_avg = dataset.filt.rolling(window_size, min_periods=1).mean()

# Process moving average
mov_avg = [x * 1.2 for x in mov_avg]  # Apply scaling
dataset['filt_rollingmean'] = mov_avg

# Mark regions of interest
window = []
peaklist = []
listpos = 0

for datapoint in dataset.filt:
    if listpos >= len(dataset.filt_rollingmean):
        break
    
    rollingmean = dataset.filt_rollingmean[listpos]
    
    if (datapoint < rollingmean) and (len(window) < 1):
        listpos += 1
    elif datapoint > rollingmean:
        window.append(datapoint)
        listpos += 1
    else:
        if window:  # Check if window is not empty
            maximum = max(window)
            beatposition = listpos - len(window) + window.index(maximum)
            peaklist.append(beatposition)
        window = []
        listpos += 1

# Plot peak detection
fig_hr = plt.figure(num='Peak detector')
ax5 = fig_hr.add_subplot(111)
ax5.set_title("Detected peaks in signal")
ax5.plot(dataset.filt, alpha=0.5, color='blue')
ax5.plot(mov_avg, color='green')
ax5.scatter(peaklist, [dataset.filt[x] for x in peaklist], color='red')

# Compute heart rate
RR_list = []
for i in range(1, len(peaklist)):
    RR_interval = peaklist[i] - peaklist[i-1]
    ms_dist = (RR_interval / fs) * 1000.0
    RR_list.append(ms_dist)

if RR_list:
    bpm = 60000 / np.mean(RR_list)
    print(f"\nAverage Heart Beat is: {bpm:.1f}")
    print(f"No of peaks in sample: {len(peaklist)}")
else:
    print("No peaks detected")

plt.show()
filtered_ecg_data = pd.DataFrame({
    'hart': y_filt  # Use correct column name
})

# Save to CSV with header
filtered_ecg_data.to_csv('filtered_ecg.csv', index=False)
print("\nFiltered ECG data saved to 'filtered_ecg.csv' with 'hart' header")

plt.show()