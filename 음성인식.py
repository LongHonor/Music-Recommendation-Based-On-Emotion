import torch
import torch.nn as nn

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




def recognizeVoice(tmp):
    
    import pandas as pd
    import numpy as np
    import sklearn
    import os
    from os.path import join
    import torch
    import torch.nn as nn


    emotion = ['angry','calm','disgust','fearful','happy','neutral','sad','surprise']



    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit_transform(emotion)


    import torch
    import torch.nn as nn

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
        
    model = ParallelModel(num_emotions=8).to(device)
    print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )


    path = './'
    model = torch.load(os.path.join(path,'model_300.pth'))


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


    import joblib
    from sklearn.preprocessing import StandardScaler
    file_name = 'scaler.pkl'
    scaler = joblib.load(file_name)


    path = './'
    sound = []
    hi=join(path, 'temp.wav')
    sound.append(load_audiofiles(hi))

    sound


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


    mel_sound.shape


    x_sound = np.expand_dims(mel_sound, 1)

    b,c,h,w = x_sound.shape
    x_sound = np.reshape(x_sound, newshape=(b,-1))
    x_sound = scaler.transform(x_sound)
    x_sound = np.reshape(x_sound, newshape=(b,c,h,w))


    predicts1 = []

    t = torch.Tensor(np.expand_dims(x_sound[0], 0)).to(device)
    H = model(t)
    predict1 = torch.argmax(H, 1).cpu().detach().numpy()[0]
    predicts1.append(predict1)


    predicts1


    answer = le.inverse_transform(predicts1)
    print(answer[0])
    


    import urllib.parse
    import urllib.request
    from bs4 import BeautifulSoup
    import time
    import csv

    #####웹 사이트 정보 해석 및 읽어오기#####
    hdr = { 'User-Agent' : 'Mozilla/5.0' }
    url = 'https://www.melon.com/chart/index.htm'

    req = urllib.request.Request(url, headers=hdr)
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, 'html.parser')

    #####웹 정보 가져와 담기#####
    lst50 = soup.select('.lst50, .lst100') # .xxx 는 Class 표기

    melonList = []

    cell_line = []
    genre1 = []

    for i in lst50:
        cell_line.append(i['data-song-no'])

    for j in range(0,100):
        url2 = 'https://www.melon.com/song/detail.htm?songId=' + cell_line[j]
        req2 = urllib.request.Request(url2, headers=hdr)
        html2 = urllib.request.urlopen(req2).read()
        soup2 = BeautifulSoup(html2, 'html.parser')
        attr = soup2.select_one('dl')
        genre = attr.select('dd')
        if ', ' in genre[2].text:
            if '댄스' in genre[2].text:
                genre1.append('댄스')
            elif '록/메탈' in genre[2].text:
                genre1.append('록/메탈')
            elif 'POP' in genre[2].text:
                genre1.append('POP')
            elif 'R&B/Soul' in genre[2].text:
                genre1.append('R&B/Soul')
            elif '랩/힙합' in genre[2].text:
                genre1.append('랩/힙합')
            elif '발라드' in genre[2].text:
                genre1.append('발라드')
            elif '성인가요/트로트' in genre[2].text:
                genre1.append('성인가요/트로트')
            elif '포크/블루스' in genre[2].text:
                genre1.append('포크/블루스')
            elif '재즈' in genre[2].text:
                genre1.append('재즈')
        else:
            genre1.append(genre[2].text)

    j = 0

    for i in lst50:
        temp = []
        #temp.append(i.select_one('.rank').text)  #순위 가져오기
        temp.append(i.select_one('.ellipsis.rank01').a.text) #제목 가져오기
        temp.append(i.select_one('.ellipsis.rank02').a.text) #아티스트 가져오기
        #temp.append(i.select_one('.ellipsis.rank03').a.text) #앨범 가져오기
        temp.append(genre1[j])
        melonList.append(temp)
        j+=1

        print("완료!")


    melon = pd.DataFrame(melonList)
    melonList1 = []
    for i in range (100):
        if answer[0] == 'happy':
            if melon.iloc[i,2] == '댄스':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '랩/힙합':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '성인가요/트로트':
                melonList1.append(melon.iloc[i])
        elif answer[0] == 'sad':
            if melon.iloc[i,2] == '발라드':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '재즈':
                melonList1.append(melon.iloc[i])
        elif answer[0] == 'angry':
            if melon.iloc[i,2] == '발라드':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '록/메탈':
                melonList1.append(melon.iloc[i])
        elif answer[0] == 'disgust':
            if melon.iloc[i,2] == '발라드':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '록/메탈':
                melonList1.append(melon.iloc[i])
        elif answer[0] == 'calm':
            if melon.iloc[i,2] == '댄스':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '발라드':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == 'POP':
                melonList1.append(melon.iloc[i])
        elif answer[0] == 'neutral':
            if melon.iloc[i,2] == '댄스':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '발라드':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == 'POP':
                melonList1.append(melon.iloc[i])
        elif answer[0] == 'surprised':
            if melon.iloc[i,2] == 'POP':
                melonList1.append(melon.iloc[i])
            elif melon.iloc[i,2] == '랩/힙합':
                melonList1.append(melon.iloc[i])
        elif answer[0] == 'fearful':
            if melon.iloc[i,2] == '포크/블루스':
                melonList1.append(melon.iloc[i])
                


    with open('classified_melonList.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(melonList1)
