
#######################################################################################
Developed and Tested on

OS: ubuntu 16.04
Language: Python-3.6
Deep Learning tool: Pytorch-1.7
GPU: minimum one GPU 1080 or above is required

#######################################################################################
Packages need to be installed:

numpy
scipy
pyroomacoustics
matplotlib
pyaudio
webrtcvad
librosacd 
torch 1.7
torchvision
torchaudio
scikit-learn
tqdm

#######################################################################################
To run the integrated system:

1. Connect Microphone Array to the PC or Laptop
2. run the following script in the terminal
    python intgsys_realtime.py

#######################################################################################
FOR RASPBERRY PI SETUP:

Hardware and software requirements as well as steps to install the Raspbian OS can be found here:
https://ubuntu.com/tutorials/how-to-install-ubuntu-on-your-raspberry-pi#1-overview

After installation, the pi can be used by connecting it to a monitor, mouse and keyboard.
Another alternative to using the pi is through ssh. Guide to enable ssh without a monitor can be found here:
https://phoenixnap.com/kb/enable-ssh-raspberry-pi

To connect to the raspberry pi through ssh, follow these steps:
1. Scan through the network to discover the ip address of the pi
	`$ sudo nmap -sn 192.168.0.1-255`
2. The default user and password for pi is:
	User: pi
	Password: raspberry
3. Connect to the pi using the ip address found, e.g.:
	`$ ssh pi@192.168.0.1`

To enable remote viewing, VNC server can be used by following these steps:
1. Connecting to the pi through ssh using the above instructions.
2. In pi terminal, installing tightvncserver:
	`$ sudo apt update`
	`$ sudo apt install tightvncserver`
3. In pi terminal, enabling tightvncserver access for the current machine:
	`$ tightvncserver`
4. In local machine, downloading and installing VNC viewer:
	Download: https://www.realvnc.com/en/connect/download/viewer/linux/
	`$ sudo apt install /path/to/installation/file`
5. In local machine, connecting to the pi from VNC viewer using the ip address and given PC id, e.g. 192.168.0.1:1.

**SCP commands to transfer files between rasp pi & other devices**
https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/

