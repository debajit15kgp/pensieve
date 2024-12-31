import os
import numpy as np

VIDEO_SOURCE_FOLDER = './test_videos/'  # Folder containing different video folders
VIDEO_OUTPUT_FOLDER = './test_video/'
BITRATE_LEVELS = 6
MASK = [0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
M_IN_B = 1000000.0

# Keep track of output file number
output_file_num = 0

# Get all folders
video_folders = sorted([d for d in os.listdir(VIDEO_SOURCE_FOLDER) 
                       if os.path.isdir(os.path.join(VIDEO_SOURCE_FOLDER, d))])

for folder in video_folders:
    folder_path = os.path.join(VIDEO_SOURCE_FOLDER, folder)
    
    # Get all m4s files in this folder
    m4s_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.m4s')])
    
    # Process each m4s file as a separate video
    for m4s_file in m4s_files:
        video_chunk_sizes = []
        
        for bitrate in xrange(BITRATE_LEVELS):
            chunk_path = os.path.join(folder_path, m4s_file)
            if os.path.exists(chunk_path):
                chunk_size = os.path.getsize(chunk_path) / M_IN_B
                video_chunk_sizes.append(chunk_size)
            else:
                video_chunk_sizes.append(0)  # or some default value
                
        # Create output file with unique number
        output_path = os.path.join(VIDEO_OUTPUT_FOLDER, str(output_file_num) + '.y4m')
        with open(output_path, 'wb') as f:
            # Write single chunk video format
            f.write(str(BITRATE_LEVELS) + '\t1\n')  # Single chunk per video
            for m in MASK:
                f.write(str(m) + '\t')
            f.write('\n')
            # Write the chunk sizes for all bitrates
            for size in video_chunk_sizes:
                f.write(str(size) + '\t')
            f.write('\n')
        
        print("Created output file: {} from {}".format(output_path, m4s_file))
        output_file_num += 1

print("Finished processing {} individual videos".format(output_file_num))