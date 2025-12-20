import os
import cv2
from tqdm import tqdm
import subprocess
import imageio

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

# def create_video_from_frames(rgbs, output_video_path, fps, audio_path=None):
#     temp_video_path = output_video_path.replace('.mp4', '_temp_no_audio.mp4')
#     h, w, _ = rgbs[0].shape
#     size = (w, h)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(temp_video_path, fourcc, fps, size)
#     for frame_rgb in tqdm(rgbs):
#         out.write(frame_rgb)
#     out.release()
#     add_audio_to_video(temp_video_path, audio_path, output_video_path)


# def create_video_from_frames(rgbs, output_video_path, fps, audio_path=None):
#     # 1. 设置视频临时路径
#     temp_video_path = output_video_path.replace('.mp4', '_temp_no_audio.mp4')
    
#     # 2. 设置 GIF 输出路径 (同名，同目录，后缀改为 .gif)
#     output_gif_path = output_video_path.replace('.mp4', '.gif')

#     h, w, _ = rgbs[0].shape
#     size = (w, h)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(temp_video_path, fourcc, fps, size)
    
#     # 用于存储转换为 RGB 格式的帧列表 (OpenCV默认是BGR，GIF需要RGB)
#     gif_frames = []

#     print(f"正在生成视频和 GIF...")
#     for frame_bgr in tqdm(rgbs):
#         # A. 写入 MP4 视频 (OpenCV 使用 BGR)
#         out.write(frame_bgr)
        
#         # B. 准备 GIF 帧 (BGR -> RGB)
#         # 如果你的 rgbs 列表本身就是 RGB 格式，这就不用转，直接 append frame_bgr 即可
#         # 但通常 cv2 读取或处理的都是 BGR，所以这里默认做一次转换
#         frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#         gif_frames.append(frame_rgb)

#     out.release()

#     # 3. 保存 GIF 文件
#     # loop=0 表示无限循环播放
#     print(f"正在保存 GIF 到: {output_gif_path}")
#     imageio.mimsave(output_gif_path, gif_frames, fps=fps, loop=0)

#     # 4. 处理音频 (原逻辑)
#     add_audio_to_video(temp_video_path, audio_path, output_video_path)

def create_video_from_frames(rgbs, output_video_path, fps, audio_path=None):
    """
    生成 MP4 视频，并同时生成同名 GIF。
    """
    if not rgbs: return

    # 1. 路径设置
    temp_video_path = output_video_path.replace('.mp4', '_temp_no_audio.mp4')
    output_gif_path = output_video_path.replace('.mp4', '.gif')

    h, w, _ = rgbs[0].shape
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, size)
    
    gif_frames = []

    print(f"正在生成视频和 GIF (FPS={fps})...")
    # 为了防止内存爆炸，如果视频很长，我们可以对 GIF 进行降采样（可选）
    # 这里保持全帧率全分辨率
    for frame_bgr in tqdm(rgbs):
        # 写入 MP4
        out.write(frame_bgr)
        
        # 准备 GIF (BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)

    out.release()

    # 3. 保存 GIF
    print(f"正在保存 GIF 到: {output_gif_path}")
    try:
        # imageio 可能会很慢或占内存，如果帧数太多建议 fps/2 或 resize
        imageio.mimsave(output_gif_path, gif_frames, fps=fps, loop=0)
    except Exception as e:
        print(f"GIF 保存失败: {e}")

    # 4. 合并音频
    add_audio_to_video(temp_video_path, audio_path, output_video_path)