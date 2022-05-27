from os import link
import tkinter
import webbrowser
import recording as rc
#import sth
import createPlaylist as cp

def callback(url):
    webbrowser.open_new(url)

def changeColor(button, color):
    button.configure(bg = color)

def recording(rc, button):
    changeColor(button,'red')
    rc.startRecording()
    changeColor(button,'white')

window = tkinter.Tk()
window.title = "Music Recommendation by Emotion"
window.geometry("640x400+100+100")
window.resizable(False, False)

recorder = rc.recorder()
linkCreater = cp.playlistCreater()

startButton = tkinter.Button(text="start", bg = 'white')
startButton.pack()
startButton.bind("<Button-1>", recorder.startRecording)

makelistButton = tkinter.Button()
makelistButton.pack()
makelistButton.bind("<Button-1>",)

# stopButton = tkinter.Button(text="stop")
# stopButton.pack()
# stopButton.bind("<Button-1>", recorder.stopRecording)

playlistButton = tkinter.Button(text="create playlist")
playlistButton.pack()
playlistButton.bind("<Button-1>",linkCreater.resultingPlaylist)

ytLink = tkinter.Label(window, text = "Youtube Hyperlink", fg = "blue", cursor="hand2")
ytLink.pack()
ytLink.bind("<Button-1>", lambda e: callback(linkCreater.playlink))

window.mainloop()