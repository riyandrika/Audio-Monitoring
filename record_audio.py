import pyaudio
import numpy as np
import time
import wave
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_name', default='test', help='input file name')
parser.add_argument('-d', '--duration', default=60, help='recording duration')
args = parser.parse_args()

def audio_record(address_raw, rec_dur, fs=16000, no_channel=1):
    save_address = address_raw

    FORMAT = pyaudio.paInt16

    CHUNK = 128
    audio = pyaudio.PyAudio()
    channel = no_channel

    # start Recording
    stream = audio.open(format=FORMAT, channels=channel,
                        rate=fs, input=True,
                        frames_per_buffer=CHUNK)
    print(f"recording to {address_raw} for {rec_dur}s...")
    frames = []
    start_time = time.time()
    iterations = np.floor(fs / CHUNK) * rec_dur
    for i in tqdm(range(0, int(iterations))):
        data = stream.read(CHUNK)
        frames.append(data)
    print("--- %s seconds --- audio procssing time " % (time.time() - start_time))
    # print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(save_address, 'wb')
    waveFile.setnchannels(channel)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(fs)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return None

if __name__ == '__main__':
    name = str(args.input_name)
    name += '.wav'
    name = os.path.join('./recordings/', name)
    dur = int(args.duration)

    audio_record(name,dur)