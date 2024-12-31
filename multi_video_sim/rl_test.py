#!/usr/bin/env python2
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import a3c
import env

S_INFO = 7
S_LEN = 10
A_DIM = 10
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [200,300,450,750,1200,1850,2850,4300,6000,8000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
M_IN_B = 1000000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
RAND_RANGE = 1000
NN_MODEL = "../model/nn_model_ep_77400.ckpt"

# Add log file handling
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './cooked_test_traces/'
TEST_VIDEO_FOLDER = './test_video/'
NETWORK_LOG_FILE = './test_results/network_metrics.csv'

def setup_logging():
    if not os.path.exists('./test_network_results_all_traces/'):
        os.makedirs('./test_network_results_all_traces/')
    
    # with open(NETWORK_LOG_FILE, 'w') as f:
    #     f.write('timestamp,size,ssim_index,cwnd,in_flight,min_rtt,rtt,delivery_rate,buffer,cum_rebuf,packet_drop,packet_drop_rate\n')

def log_metrics(log_file, timestamp, bit_rate, buffer_size, rebuf, video_chunk_size, delay, reward):
    log_file.write(str(timestamp / M_IN_K) + '\t' +
                   str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                   str(buffer_size) + '\t' +
                   str(rebuf) + '\t' +
                   str(video_chunk_size) + '\t' +
                   str(delay) + '\t' +
                   str(reward) + '\n')
    log_file.flush()

def bitrate_to_action(bitrate, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert bitrate >= 0
    assert bitrate < np.sum(mask)
    cumsum_mask = np.cumsum(mask) - 1
    action = np.where(cumsum_mask == bitrate)[0][0]
    return action

def main():
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    setup_logging()

    # Initialize environment
    net_env = env.Environment(fixed_env=True,
                            trace_folder=TEST_TRACES,
                            video_folder=TEST_VIDEO_FOLDER)

    # Count total test videos
    # total_videos = 1
    total_videos = len([f for f in os.listdir(TEST_VIDEO_FOLDER) if not f.startswith('.')])
    total_traces = len(net_env.all_cooked_time)
    print "Total videos to test:", total_videos
    print "Total network traces:", total_traces

    with tf.Session() as sess:
        actor = a3c.ActorNetwork(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                  state_dim=[S_INFO, S_LEN],
                                  learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if NN_MODEL is not None:
            saver.restore(sess, NN_MODEL)
            print "Testing model restored."

        video_count = 0
        
        while video_count < total_videos:
            for trace_idx in range(total_traces):
                # Randomly set video and trace indices
                net_env.video_idx = video_count  # Process videos in order
                net_env.trace_idx = trace_idx

                time_stamp = 0
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                action = bitrate_to_action(bit_rate, net_env.video_masks[net_env.video_idx])
                last_action = action
                s_batch = [np.zeros((S_INFO, S_LEN))]
                entropy_record = []

                log_path = LOG_FILE + '_video_' + str(video_count) + '_trace_' + str(trace_idx)
                log_file = open(log_path, 'wb')

                network_log_path = './test_network_results_all_traces/network_metrics_video_' + str(video_count) + '_trace_' + str(trace_idx) + '.csv'
                with open(network_log_path, 'w') as f:
                    f.write('timestamp,size,ssim_index,cwnd,in_flight,min_rtt,rtt,delivery_rate,buffer,cum_rebuf,packet_drop,packet_drop_rate\n')

                while True:  # Process current video
                    delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, end_of_video, \
                    video_chunk_remain, video_num_chunks, \
                    next_video_chunk_sizes, mask, throughput, min_rtt, \
                    avg_rtt, avg_cwnd, avg_in_flight, delivery_rate, packet_drops, total_packets = \
                        net_env.get_video_chunk_testing(bit_rate)

                    time_stamp += delay + sleep_time

                    reward = VIDEO_BIT_RATE[action] / M_IN_K \
                            - REBUF_PENALTY * rebuf \
                            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] -
                                                    VIDEO_BIT_RATE[last_action]) / M_IN_K

                    # Log traditional metrics
                    log_metrics(log_file, time_stamp, action, buffer_size, rebuf,
                            video_chunk_size, delay, reward)

                    # Log network metrics
                    metrics = {
                        'size': video_chunk_size,
                        'ssim_index': VIDEO_BIT_RATE[action] / M_IN_K,
                        'cwnd': avg_cwnd,
                        'in_flight': avg_in_flight,
                        'min_rtt': min_rtt,
                        'rtt': avg_rtt,
                        'delivery_rate': delivery_rate,
                        'buffer': buffer_size,
                        'cum_rebuf': rebuf,
                        'packet_drops': packet_drops,
                        'packet_drop_rate': packet_drops/float(total_packets) if total_packets > 0 else 0,
                    }
                    
                    with open(network_log_path, 'a') as f:
                        f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            time_stamp,
                            metrics['size'],
                            metrics['ssim_index'],
                            metrics['cwnd'],
                            metrics['in_flight'],
                            metrics['min_rtt'],
                            metrics['rtt'],
                            metrics['delivery_rate'],
                            metrics['buffer'],
                            metrics['cum_rebuf'],
                            metrics['packet_drops'],
                            metrics['packet_drop_rate']
                        ))

                    last_bit_rate = bit_rate
                    last_action = action

                    # retrieve previous state
                    if len(s_batch) == 0:
                        state = [np.zeros((S_INFO, S_LEN))]
                    else:
                        state = np.array(s_batch[-1], copy=True)

                    # dequeue history record
                    state = np.roll(state, -1, axis=1)
                    # print("check state: ", state.shape)

                    state[0, -1] = VIDEO_BIT_RATE[action] / float(np.max(VIDEO_BIT_RATE))
                    state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
                    state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
                    state[3, -1] = float(delay) / M_IN_K
                    state[4, -1] = video_chunk_remain / float(video_num_chunks)
                    state[5, :] = -1
                    # print("state: ", state.shape)
                    # print("timestamp: ", time_stamp)
                    nxt_chnk_cnt = 0
                    for i in xrange(A_DIM):
                        if mask[i] == 1:
                            state[5, i] = next_video_chunk_sizes[nxt_chnk_cnt] / M_IN_B
                            nxt_chnk_cnt += 1
                    assert(nxt_chnk_cnt) == np.sum(mask)
                    state[6, -A_DIM:] = mask

                    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                    assert len(action_prob[0]) == np.sum(mask)
                    
                    action_cumsum = np.cumsum(action_prob)
                    bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                    action = bitrate_to_action(bit_rate, mask)

                    s_batch.append(state)
                    entropy_record.append(a3c.compute_entropy(action_prob[0]))

                    if end_of_video:
                        log_file.write('\n')
                        log_file.close()
                        print "Finished video", video_count, "with trace", trace_idx
                        break

            video_count += 1

        print "Completed testing all videos"

if __name__ == '__main__':
    main()