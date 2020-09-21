import librosa
from scipy.io.wavfile import write
import numpy as np
EPS = 1e-8

# gaussian noise if true
noise_gaussian = True

# make signal and noise
root = '/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/'
sound_file = root + "dataset/2speakers/wav8k/min/cv/s1/40oo0312_0.92357_02bc0218_-0.92357.wav"
noise_file = root + "dataset/2speakers/wav8k/min/cv/s1/01va010d_0.11813_029a010y_-0.11813.wav"
sound, _ = librosa.load(sound_file, sr=None)
noise, _ = librosa.load(noise_file, sr=None)
len_noise = 6000
noise = noise[:len_noise]
num_copies = [len(sound)//len_noise + 1]
noise = np.repeat(noise[:, np.newaxis], num_copies, axis=1).T.flatten()[:len(sound)]
if noise_gaussian:
    noise = np.random.normal(scale=noise.std(), size=noise.shape)
write(root + 'examples/' + 'sound.wav', 8000, sound)
write(root + 'examples/' + 'noise.wav', 8000, noise)

# compute SNR
# suppose noise is estimated signal, sound is target
s_hat = noise
s = sound
s_hat -= s_hat.mean()
s -= s.mean()
# projection
s_target = np.dot(s_hat, s) * s / np.sum(s ** 2)
e_noise = s_hat - s_target
si_snr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + EPS) + EPS)
print(si_snr)

