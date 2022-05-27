#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import os
from os.path import join
import torch
import torch.nn as nn


# In[2]:


emotion = ['angry','calm','disgust','fearful','happy','neutral','sad','surprise']


# In[3]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(emotion)


# In[4]:


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
        
        

        # Linear softmax layer
        
        

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


# In[5]:


path = r'C:\Users\arz61\Documents\오픈소스'
model = torch.load(os.path.join(path,'model_300.pth'))


# In[6]:


path = 'C:/Users/arz61/OneDrive/바탕 화면/4학년 1학기 홍혁기/오픈소스SW개론/melon chart/음성인식을통한 감정분류'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')] ## 파일명 끝이 .csv인 경우


# In[7]:


pd_sample = pd.read_csv(path + '/' +file_list_py[0])
pd_test = pd.read_csv(path + '/' +file_list_py[1])
pd_train = pd.read_csv(path + '/' +file_list_py[2])


# In[8]:


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


# In[9]:


from tqdm import tqdm
def load_data(data_info, isTrain=True):
    
    PATH = "C:/Users/arz61/OneDrive/바탕 화면/4학년 1학기 홍혁기/오픈소스SW개론/melon chart/음성인식을통한 감정분류"
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

#DataFlair - Split the dataset
train_data, train_label = load_data(pd_train)
test_data = load_data(pd_test, isTrain=False)


# In[10]:


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


# In[11]:


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


# In[12]:


model.eval()

predicts = []

with torch.no_grad():
    for data in tqdm(x_test):
        ###test data를 하나씩 불러와서 학습된 모델로 추론 후 predicts array에 저장합니다.
        t = torch.Tensor(np.expand_dims(data, 0)).to(device)
        H = model(t)
        predict = torch.argmax(H, 1).cpu().detach().numpy()[0]
        predicts.append(predict)
        
print(f'predict_len:{len(predicts)}')


# In[13]:


###저장된 예측값을 위에서 사용한 label encoder를 이용해 다시 문자열로 역변환합니다.
predicts1 = le.inverse_transform(predicts)
predicts1


# In[47]:


path = 'C:/Users/arz61/OneDrive/바탕 화면/4학년 1학기 홍혁기/오픈소스SW개론/melon chart/음성인식을통한 감정분류'
sound = []
hi=join(path, 'test5.wav')
sound.append(load_audiofiles(hi))


# In[48]:


sound


# In[49]:


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

mel_sound = []
mel_spectrogram = Calculate_Melspectrogram(sound[0], sample_rate=48000)
mel_sound.append(mel_spectrogram)
mel_sound = np.stack(mel_sound,axis=0)

print(f'mel_sound:{mel_sound.shape}')


# In[50]:


mel_sound.shape


# In[51]:


from sklearn.preprocessing import StandardScaler

x_sound = np.expand_dims(mel_sound, 1)

b,c,h,w = x_sound.shape
x_sound = np.reshape(x_sound, newshape=(b,-1))
x_sound = scaler.transform(x_sound)
x_sound = np.reshape(x_sound, newshape=(b,c,h,w))


# In[52]:


predicts1 = []

t = torch.Tensor(np.expand_dims(x_sound[0], 0)).to(device)
H = model(t)
predict1 = torch.argmax(H, 1).cpu().detach().numpy()[0]
predicts1.append(predict1)


# In[53]:


predicts1


# In[57]:


answer = le.inverse_transform(predicts1)
answer[0]


# In[ ]:




