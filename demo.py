from models import *
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load MS-SincResNet model
MODEL_PATH = 'MS-SincResNet.tar'
filename = 'country.00003.wav'
state_dict = torch.load(MODEL_PATH)
model = MS_SincResNet()
model.load_state_dict(state_dict['state_dict'])
model.cuda()
model.eval()

# Read wavefile
_, data = scipy.io.wavfile.read(filename)
data = signal.resample(data, 16000 * 30)
data = data[24000:72000]


# Get spectrogram, harmonic spectrogram,
# percussive spectrogram, and Mel-spectrogram
D = librosa.stft(data, n_fft=512, hop_length=128)
rp = np.max(np.abs(D))
D_harmonic, D_percussive = librosa.decompose.hpss(D)
plt.plot()
# librosa.display.specshow(np.abs(D))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp))
plt.jet()
plt.show()
plt.plot()
# librosa.display.specshow(np.abs(D_harmonic))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp))
plt.jet()
plt.show()
plt.plot()
# librosa.display.specshow(np.abs(D_percussive))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp))
plt.jet()
plt.show()




# Get the leared 2D representations from MS-SincResNet
data = torch.from_numpy(data).float()
data.unsqueeze_(dim=0)
data.unsqueeze_(dim=0)
data = data.cuda()
_, feat1, feat2, feat3 = model(data)
feat1.squeeze_()
feat2.squeeze_()
feat3.squeeze_()
feat1 = feat1.detach().cpu().numpy()
feat2 = feat2.detach().cpu().numpy()
feat3 = feat3.detach().cpu().numpy()

librosa.display.specshow(librosa.amplitude_to_db(np.abs(feat1), ref=rp))
plt.jet()
plt.show()
librosa.display.specshow(librosa.amplitude_to_db(np.abs(feat2), ref=rp))
plt.jet()
plt.show()
librosa.display.specshow(librosa.amplitude_to_db(np.abs(feat3), ref=rp))
plt.jet()
plt.show()
