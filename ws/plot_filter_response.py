import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

sos = scipy.signal.butter(5, 0.05, btype="highpass", fs=100, output="sos")
b, a = scipy.signal.butter(5, 0.05, btype="highpass", fs=100, output="ba")
w_sos, h_sos = scipy.signal.sosfreqz(sos, worN=2000, fs=100)
w_ba, h_ba = scipy.signal.freqz(b, a, worN=2000, fs=100)


def plot_response(w, h, title):
    plt.figure()
    plt.subplot(2, 1, 1)
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w, db)
    plt.ylim(-75, 5)
    plt.xlim(0, 0.25)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel("Gain [dB]")
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h))
    plt.grid(True)
    plt.yticks(
        [-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi],
        [r"$-\pi$", r"$-\pi/2$  ", "0", r"$\pi/2$", r"$\pi$"],
    )
    plt.xlim(0, 0.1)
    plt.ylabel("Phase [rad]")
    plt.xlabel("Frequency [Hz]")
    plt.show()


plot_response(w_sos, h_sos, "Frequency response SOS")
plot_response(w_ba, h_ba, "Frequency response BA")
