import json
import subprocess
import os

'''


ffprobe -v quiet -print_format json -show_format "input.mp4" > metadata.json


'''

# Load the modified metadata from the JSON file
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Define input and temporary output file paths
input_video = 'test.mkv'
temp_output_video = 'temp_output.mkv'

# Prepare the ffmpeg command
ffmpeg_command = ['ffmpeg', '-i', input_video]

# Add metadata key-value pairs from the JSON
for key, value in metadata['format']['tags'].items():
    ffmpeg_command.extend(['-metadata', f'{key}={value}'])

ffmpeg_command.extend(['-codec', 'copy', temp_output_video])

# Run the ffmpeg command to update metadata
subprocess.run(ffmpeg_command)

# Replace the original file with the new file
os.replace(temp_output_video, input_video)
