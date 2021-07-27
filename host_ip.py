import netifaces as ni
ip = ni.ifaddresses('eno1')[ni.AF_INET][0]['addr']
with open("host_ip.txt", "w") as file:
    file.write(ip)
print(ip)