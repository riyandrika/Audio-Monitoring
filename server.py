import socket
import wave
import pyaudio
import time

PORT = 8080
CHUNK = 16000*16
RATE = 16000
RECORD_SECONDS = 60
WAVE_OUTPUT_FILENAME = "Rx_8Ch_16kHz_16kb_60s.wav"
CHANNELS = 8
FORMAT = pyaudio.paInt16


server_ip = socket.gethostbyname(socket.gethostname())
print("Server IP: ", server_ip)

p = pyaudio.PyAudio()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", PORT))
    print("Socket created at port", PORT)
    print("Waiting for connection at port", PORT)
    s.listen(1)
    conn, addr = s.accept()
    print(f"{addr[0]} connected to server")

    frames = []

    packet_count = 0
    print("Start receiving...")
    start_time = time.time()
    data = conn.recv(CHUNK, socket.MSG_WAITALL)   # Receive `CHUNK` bps
    while data != b'':
        packet_count += 1
        print("Packet number {packet}: Size = {length} bytes".format(packet = packet_count, length = len(data)))
        frames.append(data)
        data = conn.recv(CHUNK, socket.MSG_WAITALL)  # receive 4096 bytes at a time
    end_time = time.time()
    print("Receiving complete")
    print(f"Time elapsed: {round(end_time-start_time, 5)} seconds")

outputFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
outputFile.setnchannels(CHANNELS)  # 2 channels
outputFile.setsampwidth(p.get_sample_size(FORMAT))  # 16 bytes/sample
outputFile.setframerate(RATE)  # 44100 frames/sec
outputFile.writeframes(b''.join(frames))
outputFile.close()

print("Stream closed", end = "\n")
p.terminate()
