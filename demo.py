from models import *
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

# Load MS-SincResNet model
MODEL_PATH = 'MS-SincResNet.tar'
filename = 'blues.00001.wav'
state_dict = torch.load(MODEL_PATH)
model = MS_SincResNet()
model.load_state_dict(state_dict['state_dict'])
model.cuda()
model.eval()

# Read wavefile
_, data = scipy.io.wavfile.read(filename)
data = signal.resample(data, 16000 * 30)
data = data[24000:24000+48000]
data = torch.from_numpy(data).float()
data.unsqueeze_(dim=0)
data.unsqueeze_(dim=0)
data = data.cuda()

# Get the leared 2D representations from MS-SincResNet
_, feat1, feat2, feat3 = model(data)
feat1.squeeze_()
feat2.squeeze_()
feat3.squeeze_()
feat1 = feat1.detach().cpu().numpy()
feat2 = feat2.detach().cpu().numpy()
feat3 = feat3.detach().cpu().numpy()

f, axarr = plt.subplots(3, 1) 
plt.hot()
axarr[0].imshow(feat1[::-1, :])
axarr[1].imshow(feat2[::-1, :])
axarr[2].imshow(feat3[::-1, :])

plt.show()
