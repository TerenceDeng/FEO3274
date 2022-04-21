from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.colors as colors
from python_speech_features import mfcc,delta
from scipy.stats import zscore

woman_path="Sounds/female.wav"
music_path="Sounds/music.wav"
male_path="Sounds/male.wav"

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
    
    plt.figure(figsize=(14, 7))#20*np.log10(Sxx[1:])
    plt.pcolormesh(t, f[:f_max_idx],Sxx[:f_max_idx,:] ,norm=colors.LogNorm(vmin=Sxx[:f_max_idx,:].max()/color_min_rel, vmax=Sxx[:f_max_idx,:].max()))
    #plt.yscale("log")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("Female speech spectrogram")
    ax = plt.gca()
    an1 = ax.annotate('Voiced speech',fontsize=24,
                      xy=(0.38,370), xycoords='data',
                      xytext=(30,80), textcoords='offset points',color='white',
                      arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3",color='white'))
    an2 = ax.annotate('Unvoiced speech',fontsize=24,
                      xy=(0.23,3000), xycoords='data',
                      xytext=(70,57), textcoords='offset points',color='white',
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
    
def show_cepstrograms(color_min_rel=1e6):
    woman_fs,woman_x=read_wav(woman_path);
    music_fs,music_x=read_wav(music_path);
    male_fs,male_x=read_wav(male_path);
    
    t_end=int(len(woman_x)/woman_fs);
    features_mfcc = mfcc(woman_x, woman_fs); # default windows of 25ms and fft of 512.
    features_mfcc_norm = zscore(features_mfcc, axis=1, ddof=1) #Normalize
    t=np.linspace(0,t_end,np.shape(features_mfcc)[0]);
    plt.figure(figsize=(14, 7))#20*np.log10(Sxx[1:])
    plt.pcolormesh(t, np.arange(np.shape(features_mfcc)[1]),features_mfcc_norm.T)
    plt.ylabel('Cepstrum coefs')
    plt.xlabel('Time [sec]')
    plt.title("Female speech cepstrogram")
    plt.colorbar()
    plt.show()
    
    t_end=int(len(music_x)/music_fs);
    features_mfcc = mfcc(music_x, music_fs);
    features_mfcc_norm = zscore(features_mfcc, axis=1, ddof=1) #Normalize
    t=np.linspace(0,t_end,np.shape(features_mfcc)[0]);
    plt.figure(figsize=(14, 7))#20*np.log10(Sxx[1:])
    plt.pcolormesh(t, np.arange(np.shape(features_mfcc)[1]),features_mfcc_norm.T)
    plt.ylabel('Cepstrum coefs')
    plt.xlabel('Time [sec]')
    plt.title("Music cepstrogram")
    plt.colorbar()
    plt.show()
    
    t_end=int(len(male_x)/male_fs);
    features_mfcc = mfcc(male_x, male_fs);
    features_mfcc_norm = zscore(features_mfcc, axis=1, ddof=1) #Normalize
    t=np.linspace(0,t_end,np.shape(features_mfcc)[0]);
    plt.figure(figsize=(14, 7))#20*np.log10(Sxx[1:])
    plt.pcolormesh(t, np.arange(np.shape(features_mfcc)[1]),features_mfcc_norm.T)
    plt.ylabel('Cepstrum coefs')
    plt.xlabel('Time [sec]')
    plt.title("Male cepstrogram")
    plt.colorbar()
    plt.show()
    
def show_corr(nfft=1024):
    woman_fs,woman_x=read_wav(woman_path);
    #music_fs,music_x=read_wav(music_path);
    #male_fs,male_x=read_wav(male_path);
    f, t, Sxx = signal.spectrogram(woman_x, woman_fs,window=signal.windows.hamming(nfft),noverlap=nfft/2)
    corr_matrix_spectrogram=np.corrcoef(np.log10(Sxx));
    features_mfcc = mfcc(woman_x, woman_fs); # default windows of 25ms and fft of 512.
    features_mfcc_norm = zscore(features_mfcc, axis=1, ddof=1) #Normalize
    corr_matrix_cepstogram=np.corrcoef(features_mfcc_norm.T);
    # plt.figure()
    # plt.pcolormesh(f,f,corr_matrix_spectrogram);
    # plt.show();
    # plt.figure();
    # coefs_mfcc_n=np.arange(np.shape(features_mfcc)[1]);
    # plt.pcolormesh(coefs_mfcc_n,coefs_mfcc_n,corr_matrix_cepstogram);
    # plt.show()
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,7))
    cmap=plt.cm.bwr
    im=ax1.pcolormesh(f, f, corr_matrix_spectrogram,cmap=cmap, vmin=-1, vmax=1)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title("Correlation of log magnitudes on spectrogram frequency bins")
    coefs_mfcc_n=np.arange(np.shape(features_mfcc)[1]);
    im=ax2.pcolormesh(coefs_mfcc_n,coefs_mfcc_n,corr_matrix_cepstogram,cmap=cmap, vmin=-1, vmax=1)
    ax2.set_xlabel("MFCC")
    ax2.set_ylabel("MFCC")
    ax2.set_title("Correlation of MFCCs")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle("Comparison of correlations", fontsize=26)
    
def get_features_with_dynamics(x,fs,N=2):
    features_mfcc = mfcc(x, fs);
    features_mfcc_norm = zscore(features_mfcc, axis=1, ddof=1) #Normalize
    features_mfcc_norm_d1 = delta(features_mfcc_norm, N)
    features_mfcc_norm_d2 = delta(features_mfcc_norm_d1, N)
    final_feature_vector = np.concatenate((features_mfcc_norm,features_mfcc_norm_d1,features_mfcc_norm_d2),axis=1)
    #Each row contains all the features
    return final_feature_vector;

def test_final_feature_vector():
    woman_fs,woman_x=read_wav(woman_path);
    a=get_features_with_dynamics(woman_x,woman_fs);
if __name__ == "__main__":
    #show_osc_behaviour();
    #show_spectrograms();
    #show_cepstrograms();
    #show_corr();
    test_final_feature_vector();