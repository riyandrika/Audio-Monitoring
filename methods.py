import pyaudio
import wave

p = pyaudio.PyAudio()

class Methods():

    def __init__(self):
        pass

    def split_channels(self, data, nchannels):
        bytestreams = {}
        for channel in range(1, nchannels + 1):
            bytestreams[f"channel{channel}"] = data[channel::nchannels]
        return bytestreams

    def int16_to_string(self, dict):
        for value in dict.values():
            value = value.tostring()

    def append_frames(self, dict, bytedict, inputchannels, outputchannels):
        no_waves = int(inputchannels/outputchannels)
        for i in range(1, no_waves + 1):
            dict[f"frame{i}"] = bytedict[f"channel{2*i-1}"].extend(bytedict[f"channel{2*i}"])
        return dict

    def save_as_wave(self, filename, frame, outputchannels = 2, FORMAT = pyaudio.paInt16, RATE = 44100, ):
        outputFile = wave.open(filename, 'wb')
        outputFile.setnchannels(outputchannels)
        outputFile.setsampwidth(p.get_sample_size(FORMAT))
        outputFile.setframerate(RATE)
        outputFile.writeframes(b''.join(frame))
        outputFile.close()