import sounddevice as sd
import soundfile as sf
from tkinter import *
import numpy as np
  
#Globals
last_recording=None;
myrecording=[];  


def classify():
    classes_to_use=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_background_noise_"]
    chosen=classes_to_use[np.random.randint(len(classes_to_use))]
    T.configure(state='normal')
    T.delete("1.0","end")
    T.insert('end', "Classification: "+ chosen)
    T.configure(state='disabled')

def playback():
    data, samplerate = sf.read('temp.wav')
    sd.default.samplerate = samplerate
    sd.play(data)

def Voice_rec():
    global myrecording
    fs = 48000
    if StartStopButton['text']=="Stop recording":
        sd.stop()
        sf.write('temp.wav', myrecording, fs)
        PlaybackButton["state"] = "normal"
        ClassifyButton["state"] = "normal"
        StartStopButton["text"]="Start recording"
    else:
        T.configure(state='normal')
        T.delete("1.0","end")
        T.insert('end', "Classification: -")
        T.configure(state='disabled')
        PlaybackButton["state"] = "disabled"
        ClassifyButton["state"] = "disabled"
        maxDur = 5
        myrecording = sd.rec(int(maxDur * fs), samplerate=fs, channels=1)
        StartStopButton["text"]="Stop recording"
    return 
  
  
master = Tk()
  
Label(master, text=" Voice Recoder : "
     ).grid(row=0, sticky=W, rowspan=5)
  
  
StartStopButton = Button(master, text="Start recording", command=Voice_rec)
PlaybackButton = Button(master, text="PlaybackButton", command=playback,state="disabled")
ClassifyButton = Button(master, text="Classify", command=classify,state="disabled")
T = Text(master, height = 5, width = 52)

StartStopButton.grid(row=1,column=0)
PlaybackButton.grid(row=1,column=1)
ClassifyButton.grid(row=1,column=2)
T.configure(state='normal')
T.insert('end', "Classification: -")
T.configure(state='disabled')
T.grid(row=2,column=0)

  
mainloop()