import os
import sys
import signal
import subprocess
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from pyvirtualdisplay import Display
from time import sleep

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager

# TO RUN: download https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz
# run sudo apt-get install python-setuptools
# run sudo apt-get install xvfb
# after untar, run sudo python setup.py install
# follow directions here: https://pypi.python.org/pypi/PyVirtualDisplay to install pyvirtualdisplay

# For chrome, need chrome driver: https://code.google.com/p/selenium/wiki/ChromeDriver
# chromedriver variable should be path to the chromedriver
# the default location for firefox is /usr/bin/firefox and chrome binary is /usr/bin/google-chrome
# if they are at those locations, don't need to specify


def timeout_handler(signum, frame):
	raise Exception("Timeout")

ip = sys.argv[1]
abr_algo = sys.argv[2]
run_time = int(sys.argv[3])
process_id = sys.argv[4]
trace_file = sys.argv[5]
sleep_time = sys.argv[6]
	
if not os.path.exists('logs'):
	os.makedirs('logs')

# prevent multiple process from being synchronized
sleep(int(sleep_time))
	
# generate url
url = 'http://' + ip + '/' + 'myindex_' + abr_algo + '.html'

# timeout signal
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(run_time + 30)
	
def setup_logging(process_id):
	"""Set up logging for the script."""
	log_dir = 'logs'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	
	log_file = os.path.join(log_dir, 'video_playback_{}.log'.format(process_id))
	
	# Python 2.7 compatible logging setup
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s: %(message)s',
		handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler()
		]
	)
	return logging.getLogger(__name__)

def check_video_playback(driver):
	"""
	Check if video is playing and log details.
	Attempts multiple methods to start the video.
	"""
	try:
		# Updated video element selectors
		video_elements = [
			(By.TAG_NAME, 'video'),
			(By.CLASS_NAME, 'video-player'),
			(By.ID, 'video'),
			(By.CSS_SELECTOR, 'video.player')
		]
		
		video_element = None
		for selector_type, selector_value in video_elements:
			try:
				video_element = WebDriverWait(driver, 5).until(
					EC.presence_of_element_located((selector_type, selector_value))
				)
				break
			except TimeoutException:
				continue
		
		if not video_element:
			# logging.warning("No video element found")
			return False, {}
		
		# Multiple methods to start video
		js_scripts = [
			# Method 1: Direct play() call
			"""
			var video = arguments[0];
			video.play();
			return {
				currentTime: video.currentTime,
				paused: video.paused,
				ended: video.ended,
				readyState: video.readyState,
				duration: video.duration
			};
			""",
			
			# Method 2: Click on video element
			"""
			var video = arguments[0];
			video.click();
			video.play();
			return {
				currentTime: video.currentTime,
				paused: video.paused,
				ended: video.ended,
				readyState: video.readyState,
				duration: video.duration
			};
			""",
			
			# Method 3: Trigger play via event
			"""
			var video = arguments[0];
			var playEvent = new Event('play');
			video.dispatchEvent(playEvent);
			video.play();
			return {
				currentTime: video.currentTime,
				paused: video.paused,
				ended: video.ended,
				readyState: video.readyState,
				duration: video.duration
			};
			"""
		]
		
		# Try multiple play methods
		for script in js_scripts:
			try:
				# Try to play video
				video_details = driver.execute_script(script, video_element)
				
				# Log raw details for debugging
				logging.info("Attempted Play - Video Details: {}".format(video_details))
				
				# Short wait to allow potential play
				sleep(1)
				
				# Recheck video status
				recheck_details = driver.execute_script("""
				var video = arguments[0];
				return {
					currentTime: video.currentTime,
					paused: video.paused,
					ended: video.ended,
					readyState: video.readyState,
					duration: video.duration
				};
				""", video_element)
				
				logging.info("Recheck Video Details: {}".format(recheck_details))
				
				# Determine if video is playing
				is_playing = (
					recheck_details.get('currentTime', 0) > 0 and
					not recheck_details.get('paused', True) and
					not recheck_details.get('ended', False) and
					recheck_details.get('readyState', 0) > 2
				)
				
				if is_playing:
					return True, recheck_details
			
			except Exception as play_error:
				pass
				logging.error("Play Method Failed: {}".format(play_error))
		
		# If all play methods fail
		return False, {}
	
	except Exception as e:
		logging.error("Error checking video playback: {}".format(e))
		return False, {}

try:
	# copy over the chrome user dir
	default_chrome_user_dir = '../abr_browser_dir/chrome_data_dir'
	chrome_user_dir = '/tmp/chrome_user_dir_id_' + process_id
	os.system('rm -r ' + chrome_user_dir)
	os.system('cp -r ' + default_chrome_user_dir + ' ' + chrome_user_dir)
	
	logger = setup_logging(process_id)
	
	if abr_algo == 'RL':
		log_path = 'logs/rl_server_{}.log'.format(process_id)
		# command = 'exec python ../rl_server/rl_server_no_training.py ' + trace_file
		command = 'exec python -u ../rl_server/rl_server_no_training.py {} 2>&1 | tee {}'.format(trace_file, log_path)
		# print(command)
	elif abr_algo == 'fastMPC':
		command = 'exec python -u ../rl_server/mpc_server.py ' + trace_file
	elif abr_algo == 'robustMPC':
		command = 'exec python -u ../rl_server/robust_mpc_server.py ' + trace_file
	else:
		command = 'exec python -u ../rl_server/simple_server.py ' + abr_algo + ' ' + trace_file
	
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	sleep(2)

	logging.getLogger('pyvirtualdisplay.display').setLevel(logging.WARNING)
	
	# to not display the page in browser
	display = Display(visible=0, size=(800,600))
	display.start()
	
	# initialize chrome drive
	options=Options()
	chrome_driver = '../abr_browser_dir/chromedriver'
	options.add_argument('--user-data-dir={}'.format(chrome_user_dir))
	options.add_argument('--ignore-certificate-errors')
	# options.binary_location = '/opt/chromium/chrome'
	# options.add_argument('--autoplay-policy=no-user-gesture-required')
	# options.add_argument('--mute-audio')  # This mutes all audio
	# options.add_experimental_option('prefs', {
	# 	'profile.default_content_settings.media_stream': 1,
	# 	'profile.default_content_settings.popups': 0,
	# 	'profile.default_content_setting_values.media_stream_mic': 1,
	# 	'profile.default_content_setting_values.media_stream_camera': 1,
	# 	'profile.default_content_setting_values.automatic_downloads': 1,
	# 	'profile.default_media_settings.audio_capture_allowed': True,
	# 	'profile.default_media_settings.video_capture_allowed': True,
	# 	'profile.managed_default_content_settings.images': 1
	# })

	driver=webdriver.Chrome(chrome_driver, chrome_options=options)

	logging.info("check chrome start")
	# run chrome
	driver.set_page_load_timeout(10)
	driver.get(url)

	logging.info("check url")
	driver.execute_script("""
		document.querySelectorAll('video').forEach(video => {
			video.defaultMuted = true;
			video.muted = true;
			video.autoplay = true;
			video.play();
		});
	""")

	# body = driver.find_element_by_tag_name('body')
	# body.click()
	# sleep(1)

	# playback_checks = []
	# for check_num in range(5):
	# 	is_playing, details = check_video_playback(driver)
	# 	playback_checks.append(is_playing)
	# 	sleep(2)  # Wait between checks
	
	# # Determine overall playback status
	# overall_playback = any(playback_checks)
	# logger.info("Overall Video Playback Status: {}".format(overall_playback))

	sleep(run_time)
	
	driver.quit()
	display.stop()

	# kill abr algorithm server
	proc.send_signal(signal.SIGINT)
	# proc.kill()
	
	print 'done'
	
except Exception as e:
	try: 
		log_file.close()
		display.stop()
	except:
		pass
	try:
		driver.quit()
	except:
		pass
	try:
		proc.send_signal(signal.SIGINT)
	except:
		pass
	
	print e	

