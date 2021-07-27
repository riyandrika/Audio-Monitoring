import socket
import numpy as np
import time
import librosa
import multiprocessing as mp
import pyaudio
import torch
import torch.nn as nn
from scipy.special import softmax
import csv
from datetime import datetime
# import matplotlib.pyplot as plt
# import soundfile as sf
# from torchvision import transforms

from src.asl import Audio_ASL
from src.enhance_boost import Enhance_Boost
from src.vad import VAD
from src.auemo.models.vgg_m import VGG_M
from src.sound_level import Sound_Level
from src.auemo.au_transform_infer import Audio_Transform

def socket_connection(port):
    HOST_IP = socket.gethostbyname(socket.gethostname())
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", port))
    print("Waiting for connection at port", port)
    tcp.listen(1)
    conn, addr = tcp.accept()
    print(f"TCP --- Client {addr[0]} connected to Server {HOST_IP}")
    return conn

class Intergrated_Sys:

    def __init__(self, fs=16000, window_asl=1,
                 rec_dur=1, vad_level=2, num_src=2, no_classes=7):

        self.fs =  fs
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

    def stream_audio(self, lock1, q, event, flag, conn):

        packet = 0
        # start_time = time.time()
        while True:
            if flag.value == 1:
                data = conn.recv(int(self.fs * self.channel * 2), socket.MSG_WAITALL)
                if data == b"":
                    break
                packet += 1
                lock1.acquire()
                q.put(data)
                event.value = 1
                lock1.release()
                # time_now = time.time()
                # time_elapsed = round(time_now - start_time, 3)
                print(f"Packet {packet}")

    def raw_to_ip(self, data):
        ip = np.fromstring(data, dtype=np.int16)
        ip = np.reshape(ip, (int(self.fs * self.rec_dur), self.channel))
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
        checkpoint = torch.load(pre_trained_model_path)
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
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network = nn.DataParallel(network, device_ids=[0])

        network = network.to(device)
        #network = self.load_model('./src/auemo/model_zoo/Epoch16-ACCtensor(78.1250).pth', network)
        #network = self.load_model('./src/auemo/model_zoo/Epoch8-ACCtensor(76.9676).pth', network)
        network = self.load_model('./src/auemo/model_zoo/model_7_class.pth', network)
        return transform, network, audio_lv

    def audio_monitoring(self, lock1, q, event, flag):

        classes = ['silence', 'clapping', 'laughing', 'scream-shout', 'conversation', 'happy', 'angry']
        #classes = ['silence', 'clapping', 'laughing', 'scream-shout', 'conversation']
        transform, network, audio_lv = self.auemo_infer()

        buffer_asl = np.zeros((self.window_asl,self.channel))
        buffer_auemo = np.zeros((self.window_auemo, self.channel))

        print('finished loading model')
        flag.value = 1

        packet = 0
        now = datetime.now()
        timenow = now.strftime("%H:%M")
        csv_files = {"angle": open(f"angle_{timenow}.csv", "w"),
                     "vad": open(f"vad_{timenow}.csv", "w"),
                     "soundlevel": open(f"soundlevel_{timenow}.csv", "w"),
                     "emotion": open(f"emotion_{timenow}.csv", "w")}
        writers = {key: csv.writer(file) for key, file in csv_files.items()}
        for writer in writers.values():
            writer.writerow(["Packet", "Data"])

        while True:
            if event.value == 1:
                s_t1 = time.time()
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

                packet += 1

                print('Estimated Angle:{} voiceactivitydetection:{} Sound Level:{}'.format(estimated_posi,
                                                                        np.round(per_speech, 2),
                                                                        audio_level))
                print('Audio Emotion', auemo_output)
                writers["angle"].writerow([packet, estimated_posi])
                writers["vad"].writerow([packet, np.round(per_speech, 2)])
                writers["soundlevel"].writerow([packet, audio_level])
                writers["emotion"].writerow([packet, auemo_output])

            else:
                pass


if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    conn = socket_connection(port = 8000)
    lock1 = mp.Lock()
    q = mp.Queue()
    event = mp.Value('i', 0)
    flag = mp.Value('i', 0)



    control = Intergrated_Sys()

    process_read = mp.Process(target=control.stream_audio, args=(lock1, q, event, flag, conn))
    process_process = mp.Process(target=control.audio_monitoring, args=(lock1, q, event, flag))

    process_read.start()
    process_process.start()

    process_read.join()
    process_process.join()