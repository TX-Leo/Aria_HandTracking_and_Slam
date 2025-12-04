import os
import cv2
from tqdm import tqdm
import subprocess

def add_audio_to_video(video_path_no_audio, audio_path, output_video_path):
    if not audio_path or not os.path.exists(audio_path):
        print(f"Audio file not found at {audio_path}, skipping merge for {os.path.basename(output_video_path)}.")
        os.rename(video_path_no_audio, output_video_path)
        return

    print(f"Adding audio to {os.path.basename(output_video_path)}...")
    command = [
        'ffmpeg', '-i', video_path_no_audio, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_video_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.remove(video_path_no_audio)
        print(f"  -> Successfully created video with audio: {os.path.basename(output_video_path)}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  -> ERROR: FFmpeg failed to add audio. The video will be silent.")
        if isinstance(e, FileNotFoundError):
            print("  -> FFmpeg command not found. Please make sure FFmpeg is installed and in your system's PATH.")
        else:
            print(f"  -> FFmpeg stderr: {e.stderr.decode()}")
        os.rename(video_path_no_audio, output_video_path)

def create_video_from_frames(rgbs, output_video_path, fps, audio_path=None):
    temp_video_path = output_video_path.replace('.mp4', '_temp_no_audio.mp4')
    h, w, _ = rgbs[0].shape
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, size)
    for frame_rgb in tqdm(rgbs):
        out.write(frame_rgb)
    out.release()
    add_audio_to_video(temp_video_path, audio_path, output_video_path)