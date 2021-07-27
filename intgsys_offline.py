import sys
import numpy as np
import librosa
import multiprocessing as mp
import wave
import pyaudio
import torch
import torch.nn as nn
from scipy.special import softmax
import matplotlib.pyplot as plt
from matplotlib import animation

# import soundfile as sf
# from torchvision import transforms

from src.asl import Audio_ASL
from src.enhance_boost import Enhance_Boost
from src.vad import VAD
from src.auemo.models.vgg_m import VGG_M
from src.sound_level import Sound_Level
from src.auemo.au_transform_infer import Audio_Transform

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_path', default='./audio/s1-45-000-end.wav', help='Input file path.')

args = parser.parse_args()

class Intergrated_Sys:
    def __init__(self, input_file,
                 fs=16000, window_asl=1,
                 rec_dur=1, vad_level=2, num_src=1, no_classes=5):

        self.fs = fs
        self.window_asl = int(window_asl*fs)
        self.window_auemo = int(10*fs)
        self.rec_dur = rec_dur
        self.vad_level = vad_level
        self.num_src = num_src
        self.channel = 7
        self.aud_enhance = Enhance_Boost(ref_noise=False, ref_noise_address=None)
        self.vad_process = VAD(fs=self.fs, level=self.vad_level)
        self.aud_localise = Audio_ASL(fs=self.fs)
        # self.no_classes = 7  # when no of classes are 7
        self.no_classes = no_classes
        self.audio_file = input_file
        self.wav_file = wave.open(self.audio_file, 'rb')
        self.fs = self.wav_file.getframerate()
        print('Reading file', self.audio_file)
        print('fs:', self.fs)
        

    def record_audio(self, lock1, q, event, flag):

        FORMAT = pyaudio.paInt16
        CHUNK = 128
        # audio = pyaudio.PyAudio()
        duration = self.rec_dur
        
        while True:
            if flag.value == 1:
                data = self.wav_file.readframes(int(self.fs * duration))
                lock1.acquire()
                q.put(data)
                event.value = 1
                lock1.release()
            else:
                pass
            
    def raw_to_ip(self, data):
        ip = np.fromstring(data, dtype=np.int16)
        ip = np.reshape(ip, (int(self.fs * self.rec_dur), self.channel+1))
        ip = ip[:,:-1]
        ip = np.array(ip, dtype=np.float32)
        ip = ip / (2 ** 15)
        return ip

    def data_preprocess(self, ip, buffer):
        update_size = int(self.fs * self.rec_dur)
        temp_buff = buffer
        temp_buff = np.vstack((temp_buff, ip))
        temp_buff = temp_buff[update_size:, :]
        buffer = temp_buff
        return buffer

    def normalize(self, ip):
        ip_norm = librosa.util.normalize(ip)
        return ip_norm

    def load_model(self, pre_trained_model_path, network):
        checkpoint = torch.load(pre_trained_model_path, map_location='cpu')
        network.load_state_dict(checkpoint['model_state_dict'])
        return network

    def auemo_infer(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        para = {}
        para['fs'] = self.fs
        para['time'] = self.window_auemo
        para['n_fft'] = 1024
        para['win_length'] = int(0.05 * 16000)
        para['hop_length'] = int(0.02 * 16000)

        transform = Audio_Transform(device=device, para=para)
        audio_lv = Sound_Level(para)

        network = VGG_M(no_class=self.no_classes)

        # if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # network = nn.DataParallel(network, device_ids=[0])

        network = network.to(device)
        network = self.load_model('./src/auemo/model_zoo/Epoch16-ACCtensor(78.1250).pth', network)
        # network = self.load_model('./src/auemo/model_zoo/model_7_class.pth', network)
        return transform, network, audio_lv

    def audio_monitoring(self, lock1, q, event, flag, estimated_pos):

        # classes = ['silence', 'clapping', 'laughing', 'scream-shout', 'conversation', 'happy', 'angry']
        classes = ['silence', 'clapping', 'laughing', 'scream-shout', 'conversation']
        transform, network, audio_lv = self.auemo_infer()

        buffer_asl = np.zeros((self.window_asl,self.channel))
        buffer_auemo = np.zeros((self.window_auemo, self.channel))

        print('finished loading model')
        flag.value = 1

        while True:
            if event.value == 1:
                lock1.acquire()
                data = q.get()
                lock1.release()

                event.value = 0
                ip = self.raw_to_ip(data)

                buffer_asl = self.data_preprocess(ip, buffer_asl)
                buffer_auemo = self.data_preprocess(ip, buffer_auemo)

                x = transform.main(buffer_auemo[:,0])
                outputs = network(x)
                logit = outputs.to('cpu').detach().numpy()
                accuracy = softmax(logit)*100
                auemo_output = ['{} {}'.format(classes[i], int(accuracy[0][i]))\
                                for i in range(len(classes))]

                audio_level = audio_lv.main(buffer_asl[:,0])
                ip_norm = self.normalize(buffer_asl)
                enhance_ip1d = self.aud_enhance.enhance(ip_norm[:, 0])
                per_speech, vad_rec = self.vad_process.estimate(enhance_ip1d)

                estimated_posi = self.aud_localise.localise(ip_norm, self.num_src)
                estimated_pos.value = int(estimated_posi[0])
                
                print('Estimated Angle:{} VAD:{} Sound Level:{}'.format(estimated_posi,
                                                                        np.round(per_speech, 2),
                                                                        audio_level))
                print('Audio Emotion', auemo_output)

            else:
                pass
                    
def animate(i):
    global estimatePos2
    ax.clear()
    ax.patch.set_facecolor('black')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_theta_zero_location('N')
    angle = (estimated_pos.value) * (np.pi / 180)
    radius = 0.8
    ax.scatter(angle, radius, s=2500, color="r")
    ax.scatter(0, 0, s=2500, color="b", marker='o')

if __name__ == "__main__":
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    lock1 = mp.Lock()
    q = mp.Queue()
    event = mp.Value('i', 0)
    flag = mp.Value('i', 0)
    estimated_pos = mp.Value('i', 0)

    control = Intergrated_Sys(args.input_path)

    process_read = mp.Process(target=control.record_audio, args=(lock1, q, event, flag))
    process_process = mp.Process(target=control.audio_monitoring, args=(lock1, q, event, flag, estimated_pos))

    process_read.start()
    process_process.start()

    anim = animation.FuncAnimation(fig, animate, interval=1000, repeat=False)
    plt.show()

    process_read.join()
    process_process.join()