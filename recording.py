#pyaudio 설치 필요

import pyaudio
import wave

class recorder:
    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format = pyaudio.paInt16, channels=1, rate = 44100, input = True, frames_per_buffer = 1024)
        self.frames = []
        self.status = False

    def startRecording(self, tmp):
        print("start")
        self.status = True
        while(self.status):
            data = self.stream.read(1024)
            self.frames.append(data)

    def stopRecording(self, tmp):
        print("stop")
        if(self.status == True):
            self.status = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.saveRecording()

    def saveRecording(self):
        sound_file = wave.open("tmp.wav","wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b''.join(self.frames))
        sound_file.close()

# recorderInstance = recorder()

# try:
#     recorderInstance.startRecording("asdf")
# except KeyboardInterrupt:
#     recorderInstance.stopRecording("asdf")