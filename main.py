import tkinter
import webbrowser
import recording as rc
#import createPlaylist as cp

def callback(url):
    webbrowser.open_new(url)

window = tkinter.Tk()
window.title = "Music Recommendation by Emotion"
window.geometry("640x400+100+100")
window.resizable(False, False)

recorder = rc.recorder()

startButton = tkinter.Button(text="start")
startButton.pack()
startButton.bind("<Button-1>", recorder.startRecording)

stopButton = tkinter.Button(text="stop")
stopButton.pack()
stopButton.bind("<Button-1>", recorder.stopRecording)

ytLink = tkinter.Label(window, text = "Youtube Hyperlink", fg = "blue", cursor="hand2")
ytLink.pack()
ytLink.bind("<Button-1>", lambda e: callback("http://www.youtube.com"))

window.mainloop()