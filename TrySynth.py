import numpy as np
from Synthese import synthese
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


outputs_base = np.array([1, 1, 0.5**1, 0.5**2, 0.5**3, 0.5**4, 0.5**5, 0.5**6, 0.5**7, 0.5**8, 0.5**9, 0.5**10,
                    0.5**11, 0.5**12, 0.5**13, 0.5**14, 0.5**15])

outputs = np.stack([outputs_base for i in range(2 * 100)])


f0 = 220 * np.ones(2 * 100)

synthese(outputs, f0)

rate, data = wav.read('Test_antoine.wav')
plt.plot(data)
plt.show()


# synthese_harmonique(outputs, f0)