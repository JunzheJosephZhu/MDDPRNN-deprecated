"""
Author: Junzhe Zhu
This script uses sph2pipe to turn all the wv1 files into wav files. Apparently wv1 and wv2 files are the same, just recorded with different mics
This script uses multiprocessing
"""
from IPython.display import Audio
from glob import glob
from tqdm import tqdm
import librosa
import os

sphfiles = list(glob("/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/egs/**/*.wv1", recursive = True))
sphfiles.sort()

def sph2wav(i):
    """
    args:
        i: index in sphfile, for multiprocessing
    This function first turns sphfiles[i] into a wavfile, and resamples and overwrites the original wavfile
    """
    sphfile = sphfiles[i]
#    sphfile = os.path.abspath(sphfile)
    wavfile = sphfile.replace('wv1', 'wav') # wavfile has same filename with sphfiles except for extension, and are in the same folder
    command = 'sph2pipe -f rif %s %s' % (sphfile, wavfile)
    res = os.system(command)
    if res: # There's like this one file that gives premature EOF
        print(command)
        print(res)
        raise RuntimeError('File is corrupt')
    sound, sr = librosa.load(wavfile, sr=8000) # this function automatically resamples the wavfiles
    librosa.output.write_wav(wavfile, sound, sr, norm = True)

from multiprocessing import Pool
pool = Pool(processes=25)   # change this depending on the number of CPUs
pool.map(sph2wav, range(0, len(sphfiles)))
print('Done!')
