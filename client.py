import socket
import time
import wave
import pyaudio
import threading

with open("host_ip.txt", "r") as file:
    HOST = file.read()
    PORT = 8000
    ADDRESS = (HOST, PORT)

CHUNK = 16000
RATE = 16000
CHANNELS = 7
FORMAT = pyaudio.paInt16
RECORD_TIME = 10

p = pyaudio.PyAudio()

tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp.connect(ADDRESS)
print("Client connected to {HOST} via TCP".format(HOST = HOST))


def detect_input_index():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    mic_indexes = []
    for i in range(0, numdevices):
        if "micArray" in p.get_device_info_by_index(i)["name"]:
            mic_indexes.append(i)
    return mic_indexes


indexes = detect_input_index()
print("Input device indexes: {indexes}".format(indexes = indexes))
index = indexes[0]


class MicInput:

    def __init__(self, index, format = FORMAT, rate = RATE, channels = CHANNELS):
        self.index = index
        self.format = format
        self.rate = rate
        self.channels = channels

    def stream(self):
        mic_stream = p.open(
                    format = self.format,
                    channels = self.channels,
                    rate = self.rate,
                    input = True,
                    input_device_index = self.index)
        mic_stream.start_stream()
        return mic_stream

    def frame(self):
        self.own_frame = []
        return self.own_frame


def Tx_TCP(index, s, record_time = RECORD_TIME):  # SENDING IN CHUNK
    mic = MicInput(index = index)
    frames = mic.frame()
    stream = mic.stream()
    start_time = time.time()
    packet = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        packet += 1
        s.send(data)
        time_now = time.time()
        time_elapsed = round(time_now - start_time, 3)
        print("Packet %d" % packet)


tcp_thread = threading.Thread(target = Tx_TCP, args = (index, tcp))
tcp_thread.start()