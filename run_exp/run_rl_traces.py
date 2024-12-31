import os
import time
import json
import urllib
import subprocess

TRACE_PATH = '../cooked_mahimahi_traces/'

with open('./chrome_retry_log', 'wb') as f:
    f.write('chrome retry log\n')

os.system('sudo sysctl -w net.ipv4.ip_forward=1')

# ip_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
# ip = str(ip_data['ip'])
ip_data = json.loads(urllib.urlopen("https://api.ipify.org?format=json").read())
ip = str(ip_data['ip'])
# ip = 'localhost' 

# Only running RL algorithm
ABR_ALGO = 'RL'
PROCESS_ID = 0  # Changed to 0 since we're only running one process
command_RL = 'python -u run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

print(command_RL)
# Launch only the RL process
proc_RL = subprocess.Popen(command_RL, stdout=subprocess.PIPE, shell=True)

stdout, stderr = proc_RL.communicate()
print("Output:", stdout)
print("Errors:", stderr)


# Wait for the RL process to complete
proc_RL.wait()