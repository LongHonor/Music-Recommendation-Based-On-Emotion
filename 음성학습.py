import pandas as pd
import numpy as np
import sklearn
import os
from os.path import join


path = './'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]


file_list_py


pd_sample = pd.read_csv(path + '/' +file_list_py[0])
pd_test = pd.read_csv(path + '/' +file_list_py[1])
pd_train = pd.read_csv(path + '/' +file_list_py[2])


pd_train


import librosa
import glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import librosa, librosa.display 

def load_audiofiles(file_name, sample_rate=48000):
    
    result=np.array([])
    
    audio_signal, sample_rate = librosa.load(file_name, duration=3, offset=0.5, sr=sample_rate)

    signal = np.zeros(int(sample_rate*3,))
    signal[:len(audio_signal)] = audio_signal
    
    return signal



from tqdm import tqdm
def load_data(data_info, isTrain=True):
    
    PATH = "./"
    if isTrain:
        train_data = []#음성 feature들을 담는 dictionary
        train_label = []#학습에 사용할 label을 담는 list
        
        file_list = data_info['file_name']
        emotion_list = data_info['emotion']
        for file_name, emotion in tqdm(zip(file_list, emotion_list)):
            
            hi=join(PATH, 'train_data',file_name)
            train_data.append(load_audiofiles(hi))
            train_label.append(emotion)
            
        return np.array(train_data), np.array(train_label)
    
    else:
        test_data = []
        file_list = data_info['file_name']
    
        for file_name in tqdm(file_list):

            hi=join(PATH, 'test_data',file_name)
            test_data.append(load_audiofiles(hi))
            
        return np.array(test_data)

train_data, train_label = load_data(pd_train)
test_data = load_data(pd_test, isTrain=False)


train_data.shape


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_label)
y_train = le.transform(train_label)


def Calculate_Melspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length = 512,
                                              window='hamming',
                                              hop_length = 256,
                                              n_mels=128,
                                              fmax=sample_rate/2
                                             )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

mel_train = []
print("Calculate mel spectrograms for train set")
train_data = np.stack(np.array(train_data),0)
test_data = np.stack(np.array(test_data),0)
for i in range(train_data.shape[0]):
    mel_spectrogram = Calculate_Melspectrogram(train_data[i,:], sample_rate=48000)
    mel_train.append(mel_spectrogram)
    print("\r Processed {}/{} files".format(i+1,train_data.shape[0]),end='')
    
print('')
mel_train = np.stack(mel_train,axis=0)

mel_test = []
for i in range(test_data.shape[0]):
    mel_spectrogram = Calculate_Melspectrogram(test_data[i,:], sample_rate=48000)
    mel_test.append(mel_spectrogram)
    print("\r Processed {}/{} files".format(i+1,test_data.shape[0]),end='')
    
print('')
mel_test = np.stack(mel_test,axis=0)

print(f'mel_train:{mel_train.shape}, mel_test:{mel_test.shape}')


mel_test[0].shape


mel_test[0]


from sklearn.preprocessing import StandardScaler

x_train = np.expand_dims(mel_train, 1) #DataNum, 1ch, H, W
x_test = np.expand_dims(mel_test, 1)

scaler = StandardScaler()

b,c,h,w = x_train.shape
x_train = np.reshape(x_train, newshape=(b,-1))
x_train = scaler.fit_transform(x_train)
x_train = np.reshape(x_train, newshape=(b,c,h,w))

b,c,h,w = x_test.shape
x_test = np.reshape(x_test, newshape=(b,-1))
x_test = scaler.transform(x_test)
x_test = np.reshape(x_test, newshape=(b,c,h,w))


print(b,c,h,w)


x_test[0].shape

x_test[0]


import torch
import torch.nn as nn

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ParallelModel(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
       
        
       # 1. 1stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2,padding=0),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(p=0.3)
        )
                    
        
        
        
        # 2. 2stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,4), stride=4,padding=0),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(p=0.3)
        )
        
        
        

        # 3. 3stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,4), stride=4,padding=0),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(p=0.3)
        )
        
        
        

        # 4. 4stage (Conv + BatchNorm + ReLU + Maxpooling + Dropout)
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,4), stride=4,padding=0),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(p=0.3)
        )
        
        self.fc = nn.Linear(1*4*128,8,bias=True)
        nn.init.orthogonal_(self.fc.weight)

    def forward(self,x):
    ##Forward
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output
    
model = ParallelModel(num_emotions=8).to(device)
print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )


get_ipython().system('pip install livelossplot')


EPOCHS=300
DATASET_SIZE = x_train.shape[0]
BATCH_SIZE = 64

OPTIMIZER = torch.optim.SGD(model.parameters(),lr=1e-2, momentum=0.9)


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


from livelossplot import PlotLosses
from tqdm import trange
liveloss = PlotLosses()

logs = {}

model.train()

for epoch in trange(EPOCHS):
    # shuffle data
    ind = np.random.permutation(DATASET_SIZE)

    x_train = x_train[ind]
    y_train = y_train[ind]
    
    epoch_loss = 0
    
    iters = int(DATASET_SIZE / BATCH_SIZE)
     
    for i in range(iters):

        batch_start = i * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
        actual_batch_size = batch_end-batch_start
        
        x = x_train[batch_start:batch_end]
        y = y_train[batch_start:batch_end]
        
        X = torch.FloatTensor(x).to(device)
        Y = torch.LongTensor(y).to(device)
        
        model.train()
        hypothesis = model(X)
        loss = loss_fnc(hypothesis, Y)
        
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        
        epoch_loss += loss.item()*actual_batch_size/DATASET_SIZE

    logs['train_loss'] = epoch_loss
        
    liveloss.update(logs)
    liveloss.draw()



model


path = './'
torch.save(model,os.path.join(path,'model_300.pth'))
