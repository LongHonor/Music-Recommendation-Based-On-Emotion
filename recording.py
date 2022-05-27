#pyaudio 설치 필요

import pyaudio
import wave
import time

class recorder:
    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format = pyaudio.paInt16, channels=1, rate = 44100, input = True, frames_per_buffer = 1024)
        self.frames = []
        self.status = False
        self.titlePre = "temp"
        self.title = ""

    def startRecording(self, tmp):
        print("start")
        self.__init__()
        self.title = self.titlePre + ".wav"
        self.status = True
        start_time = time.time()
        end_time = start_time
        while(end_time - start_time < 5):
            data = self.stream.read(1024)
            self.frames.append(data)
            end_time = time.time()
        self.stopRecording()

    def stopRecording(self):
        print("stop")
        if(self.status == True):
            self.status = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.saveRecording()

    def saveRecording(self):
        
        sound_file = wave.open(self.title,"wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b''.join(self.frames))
        sound_file.close()

# recorderInstance = recorder()
# recorderInstance.startRecording()