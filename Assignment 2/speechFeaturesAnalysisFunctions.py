from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.colors as colors

woman_path="Sounds/female.wav"
music_path="Sounds/music.wav"

def read_wav(path):
    fs, x = wavfile.read(path)
    x=x/(2**15) #Assuming 16bit wav
    return fs,x

def plot_audio_signal(x,fs,t_init=0,t_end=1e6,title="Title"):
    n_samples=len(x);
    t_start_index=int(fs*t_init);
    t_end_index=int(fs*t_end)
    t_end_index = -1 if t_end_index>n_samples else t_end_index;
    t=np.linspace(0,n_samples/fs,n_samples);
    plt.figure();
    plt.plot(t[t_start_index:t_end_index],x[t_start_index:t_end_index])
    plt.xlabel("Time [s]")
    plt.ylabel("Signal values")
    plt.title(title)
    plt.grid()
    plt.show()    
    
def show_osc_behaviour(): #First point of the assignment
    woman_fs,woman_x=read_wav(woman_path);
    music_fs,music_x=read_wav(music_path);
    plot_audio_signal(woman_x,woman_fs,t_init=0.52,t_end=0.64,title="Woman voiced speech")
    #Here note that the frequency can be estimated and its around ..., which is a normal frequency for a female
    plot_audio_signal(woman_x,woman_fs,t_init=0.16,t_end=0.17,title="Woman unvoiced speech")
    #Here note that the output looks more like noise. However its hard to see in time, but will be easier in frequency
    plot_audio_signal(music_x,music_fs,t_init=0.075,t_end=0.11,title="Music oscilatory behaviour")
    #Not much more to say, here something very oscillating

def show_spectrograms_woman(color_min_rel=1e4,f_max=5e3,nfft=2048): 
    woman_fs,woman_x=read_wav(woman_path);
    f, t, Sxx = signal.spectrogram(woman_x, woman_fs,window=signal.windows.hamming(nfft),noverlap=nfft/2)
    f_max_idx=np.where(f>f_max)[0][0];
    plt.figure(figsize=(14, 7))#20*np.log10(Sxx[1:])
    plt.pcolormesh(t, f[:f_max_idx],Sxx[:f_max_idx,:] ,norm=colors.LogNorm(vmin=Sxx[:f_max_idx,:].max()/color_min_rel, vmax=Sxx[:f_max_idx,:].max()))
    #plt.yscale("log")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("Female speech spectrogram")
    ax = plt.gca()
    an1 = ax.annotate('Harmonics',fontsize=24,
                      xy=(0.42,370), xycoords='data',
                      xytext=(30,80), textcoords='offset points',color='white',
                      arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color='white'))
    an2 = ax.annotate('',
                      xy=(0.42,545), xycoords='data',
                      xytext=(70,57), textcoords='offset points',
                      arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color='white'))
    an3 = ax.annotate('',
                      xy=(0.42,734), xycoords='data',
                      xytext=(60,42), textcoords='offset points',
                      arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color='white'))
    plt.colorbar()
    plt.show()
def show_spectrograms_music(color_min_rel=1e6,f_max=5e3,nfft=1024): 
    music_fs,music_x=read_wav(music_path);
    f, t, Sxx = signal.spectrogram(music_x, music_fs,window=signal.windows.hamming(nfft),noverlap=nfft/2)
    f_max_idx=np.where(f>f_max)[0][0];
    plt.figure(figsize=(14, 7))#20*np.log10(Sxx[1:])
    plt.pcolormesh(t, f[:f_max_idx],Sxx[:f_max_idx,:] ,norm=colors.LogNorm(vmin=Sxx[:f_max_idx,:].max()/color_min_rel, vmax=Sxx[:f_max_idx,:].max()))
    #plt.yscale("log")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("Music spectrogram")
    ax = plt.gca()
    an1 = ax.annotate('Harmonics',fontsize=24,
                      xy=(0.15,700), xycoords='data',
                      xytext=(30,80), textcoords='offset points',color='white',
                      arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color='white'))
    an2 = ax.annotate('',
                      xy=(0.15,1000), xycoords='data',
                      xytext=(70,50), textcoords='offset points',
                      arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color='white'))
    an3 = ax.annotate('',
                      xy=(0.15,1400), xycoords='data',
                      xytext=(60,20), textcoords='offset points',
                      arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color='white'))
    plt.colorbar()
    plt.show()
def show_spectrograms():
    show_spectrograms_woman();
    show_spectrograms_music();
    
if __name__ == "__main__":
    show_osc_behaviour();
    show_spectrograms();