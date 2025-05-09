import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as signal
import os
from scipy.fftpack import fft,ifft
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import glob

segment_table = [[0,40],[40,300],[100,800],[0,1600]]

def FFT_filter(path):
	down_sample = 1
	time = 5
	sample_rate, sig = get_audio(down_sample,time,path)
	x=np.linspace(0,sample_rate*time,sample_rate*time)
	Yf = fft(sig)
	Yf = Yf[range(int(len(Yf)/2))]/sample_rate
	Xf = np.linspace(0,int(sample_rate/2),int(sample_rate*time/2))
	Yf_list = []
	Xf_list = []
	X_list = []
	Y_list = []
	for i in range(0,len(segment_table),1):
		start = segment_table[i][0]
		end = segment_table[i][1]
		Xf_list.append(np.linspace(start,end,(end-start)*time))
		Yf_list.append(abs(Yf[start*time:end*time]))
		X_list.append(np.linspace(0,time,(end-start)*time))
		Y_list.append(ifft(Yf[start*time:end*time]).astype(sig.dtype))
	# Y = ifft(Yf)
	# Y_trans = Y.astype(sig.dtype)
	# X_trans = x/sample_rate
	# return X_trans,Y_trans
	return Xf_list,Yf_list,X_list,Y_list


def plot_single_audio(file_path, segment_table, title="FFT Filter Result"):
	xf, yf, x, y = FFT_filter(file_path, segment_table)

	plt.figure(figsize=(12, len(x) * 3))
	for j in range(len(x)):
		plt.subplot(len(x), 2, j * 2 + 1)
		plt.plot(xf[j], yf[j])
		plt.xlabel("Frequency (Hz)")
		plt.ylabel("Amplitude")
		plt.title(f"Segment {j + 1} - FFT")

		plt.subplot(len(x), 2, j * 2 + 2)
		plt.plot(x[j], y[j], color='red')
		plt.xlabel("Time (s)")
		plt.ylabel("Amplitude")
		plt.title(f"Segment {j + 1} - IFFT")

	plt.tight_layout()
	plt.suptitle(title, fontsize=14, y=1.02)
	plt.show()

def FT_plot(file_path_list,Name):
	N = len(file_path_list)
	#plt.figure(Name)
	for i in range(0,N,1):
		plt.figure(file_path_list[i],figsize=(10,5))
		xf,yf,x,y = FFT_filter(file_path_list[i]) # FFt filter
		for j in range(0,len(x),1):
			plt.subplot(len(x),2,(j+1)*2)
			plt.plot(x[j],y[j],color = 'red')
			plt.xlabel("Time(s)")
			plt.subplot(len(x),2,(j+1)*2-1)
			plt.plot(xf[j],yf[j])
			plt.xlabel("Frequency(hz)")



def get_audio(down_index,time,path):
	sample_rate, sig = wavfile.read(path)
	resampleing_rate = int(sample_rate/down_index)
	down_sig = signal.resample(sig,resampleing_rate*time)
	return resampleing_rate, down_sig

def traversal_and_run(func,root_path):
	full_paths = [y for x in os.walk(root_path) for y in glob.glob(os.path.join(x[0], '*.wav'))]
	z = func(full_paths,"")


traversal_and_run(FT_plot,'validate_data/2024-10-29/leak')

plt.show()
