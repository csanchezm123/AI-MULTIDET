from pathlib import Path
import requests
from instagrapi import Client
import os
import random
import yt_dlp
from moviepy.editor import VideoFileClip


def get_image_instagram(link):
    rand = random.randint(100000, 999999)
    folder_path = "./tmp/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image_name = "image_instagram_"+str(rand)+".jpg"
    image_path = Path(folder_path+image_name)
    path = str(folder_path)+str(image_path)
    cl = Client()
    cl.login(user, pasword)
    cl.delay_range = [1, 3]
    image_id = cl.media_pk_from_url(link)
    image_info = cl.media_info(image_id)
    image_url = image_info.thumbnail_url
    response = requests.get(image_url)
    response.raise_for_status()
    with open(image_path, 'wb') as file:
        file.write(response.content)
    file.close()
    return path



def get_video_youtube(link):
    rand = random.randint(100000, 999999)
    folder_path = "./tmp/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    video_name = "video_youtube_"+str(rand)+".mp4"
    path=folder_path+video_name
    ydl_opts = {
        'outtmpl': path,
        'format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    return path


def get_audio_youtube(link):
    rand = random.randint(100000, 999999)
    folder_path = "./tmp/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    video_name = "video_youtube_"+str(rand)+".mp4"
    video_path=folder_path+video_name
    ydl_opts = {
        'outtmpl': video_path,
        'format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    audio_name = "audio_youtube_"+str(rand)+".mp3"
    path=folder_path+audio_name
    videofileclip = VideoFileClip(video_path)
    videofileclip.audio.write_audiofile(path)
    return video_path, path

