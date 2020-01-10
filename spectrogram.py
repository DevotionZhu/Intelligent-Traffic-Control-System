import os
import wave
import pylab
import struct

def graph_spectrogram(wav_file,name):
    sound_info, frame_rate = get_wav_info(wav_file)
    # By default generated spectrograms are of size 19 cm x 12 cm. This can be changed by changing the values in figsize argument
    # Eg figsize=(5,5) will generate images of size 5 cm x 5 cm

    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111) 
    pylab.axis('off')
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(name+".png",transparent=True)
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# The dataset is to be downloaded, Extracted, and content of both folders (Ambulance and Firetruck) - 
# to be kept in the same directory as this script. Then run this Script using Python

for i in range(1,201):
    graph_spectrogram("sound_"+str(i)+".png","sound_"+str(i))

for i in range(201,401):
    graph_spectrogram("sound_"+str(i)+".png","sound_"+str(i))
