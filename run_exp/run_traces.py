import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 320  # sec
MM_DELAY = 40   # millisec


def main():
	trace_path = sys.argv[1]
	abr_algo = sys.argv[2]
	process_id = sys.argv[3]
	ip = sys.argv[4]

	sleep_vec = range(1, 10)  # random sleep second

	files = os.listdir(trace_path)
	for f in files:

		while True:

			np.random.shuffle(sleep_vec)
			sleep_time = sleep_vec[int(process_id)]

			cmd = "mm-delay {0} mm-link 12mbps {1} python {2} {3} {4} {5} {6} {7} {8}".format(
				str(MM_DELAY),
				trace_path + f,
				RUN_SCRIPT,
				ip,
				abr_algo,
				str(RUN_TIME),
				process_id,
				f,
				str(sleep_time)
			)
			print(cmd)
			proc = subprocess.Popen('mm-delay ' + str(MM_DELAY) + 
					  ' mm-link 12mbps ' + trace_path + f + ' ' +
					  'python ' + RUN_SCRIPT + ' ' + ip + ' ' +
					  abr_algo + ' ' + str(RUN_TIME) + ' ' +
					  process_id + ' ' + f + ' ' + str(sleep_time),
					  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

			(out, err) = proc.communicate()

			if out == 'done\n':
				break
			else:
				with open('./chrome_retry_log', 'ab') as log:
					log.write(abr_algo + '_' + f + '\n')
					log.write(out + '\n')
					log.flush()



if __name__ == '__main__':
	main()
