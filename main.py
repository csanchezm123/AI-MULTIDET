from flask import Flask, render_template, request, redirect, url_for, after_this_request
import os
from predict import predict_normal_text, predict_news_text, predict_text_x, predict_image_cifake, predict_video_icpr, predict_video_faceforensics, predict_audio_vocoder
from langdetect import detect
from get_social_media import get_image_instagram, get_video_youtube, get_audio_youtube
from models.image.CNNdetection.predict_cnn_image import predict_image_cnn
import random
import numpy

app = Flask(__name__)

'''SELECCIÓN DE DETECCIÓN'''

#Portal principal con las 4 opciones de detección
@app.route('/')
def portal():
   return render_template('portal.html')

#Opciones de detección de textos
@app.route('/detect_text_opt')
def detect_text_opt():
   return render_template('text/detect_text_opt.html')

#Opciones de detección de imágenes
@app.route('/detect_image_opt')
def detect_image_opt():
   return render_template('image/detect_image_opt.html')

#Opciones de detección de vídeos
@app.route('/detect_video_opt')
def detect_video_opt():
   return render_template('video/detect_video_opt.html')

#Opciones de detección de audios
@app.route('/detect_audio_opt')
def detect_audio_opt():
   return render_template('audio/detect_audio_opt.html')


'''INTRODUCCIÓN O CARGA DE DATOS'''

#Introducción de texto plano
@app.route('/upload_text_manual', methods=['GET', 'POST'])
def upload_text_manual():
   return render_template('text/upload_text_manual.html')

#Introducción de texto proveniente de X
@app.route('/upload_text_x', methods=['GET', 'POST'])
def upload_text_x():
   return render_template('text/upload_text_x.html')

#Detección de imagen desde archivo
@app.route('/upload_image_manual', methods=['GET', 'POST'])
def upload_image_manual():
   return render_template('image/upload_image_manual.html')

#Detección de imagen de Instagram
@app.route('/upload_image_instagram', methods=['GET', 'POST'])
def upload_image_instagram():
   return render_template('image/upload_image_instagram.html')

#Detección de vídeo desde archivo
@app.route('/upload_video_manual', methods=['GET', 'POST'])
def upload_video_manual():
   return render_template('video/upload_video_manual.html')

#Detección de vídeo de YouTube
@app.route('/upload_video_youtube', methods=['GET', 'POST'])
def upload_video_youtube():
   return render_template('video/upload_video_youtube.html')

#Detección de audio desde archivo
@app.route('/upload_audio_manual', methods=['GET', 'POST'])
def upload_audio_manual():
   return render_template('audio/upload_audio_manual.html')

#Detección de audio de vídeo de YouTube
@app.route('/upload_audio_youtube', methods=['GET', 'POST'])
def upload_audio_youtube():
   return render_template('audio/upload_audio_youtube.html')


'''MUESTRA DE RESULTADOS'''


@app.route('/result_text_manual', methods = ['GET', 'POST'])
def result_text_manual():
   user_text = str(request.form.get('user_text', ''))
   lang_detect = detect(user_text)
   if lang_detect=="es":
      percentage_predicted_1 = "N/A"
      percentage_predicted_2 = str(predict_news_text(user_text))
      percentage_predicted_t = percentage_predicted_2
   else:
      percentage_predicted_1 = str(predict_normal_text(user_text))
      percentage_predicted_2 = "N/A"
      percentage_predicted_t = percentage_predicted_1
   return render_template('text/result_text_manual.html', percentage_predicted_1=percentage_predicted_1, percentage_predicted_2=percentage_predicted_2, percentage_predicted_t=percentage_predicted_t)


@app.route('/result_text_x', methods = ['GET', 'POST'])
def result_text_x():
   tweet_text = str(request.form.get('tweet_text', ''))
   percentage_predicted = str(predict_text_x(tweet_text))
   return render_template('text/result_text_x.html', percentage_predicted=percentage_predicted)


@app.route('/result_image_manual', methods = ['GET', 'POST'])
def result_image_manual():
   file = request.files['file']
   image_path = download_image_manual(file)
   w1=93
   w2=93
   p1=predict_image_cnn(image_path)
   p2=numpy.round(float(predict_image_cifake(image_path)), decimals=2)
   percentage_predicted_t = (p1*w1+p2*w2)/(w1+w2)
   percentage_predicted_t = numpy.round((percentage_predicted_t),decimals=2)
   os.remove(image_path)
   return render_template('image/result_image_manual.html', percentage_predicted_1=p1, percentage_predicted_2=p2, percentage_predicted_t=percentage_predicted_t)


@app.route('/result_image_instagram', methods = ['GET', 'POST'])
def result_image_instagram():
   image_link = str(request.form.get('instagram_link', ''))
   image_path = get_image_instagram(image_link)
   w1=93
   w2=93
   p1=predict_image_cnn(image_path)
   p2=numpy.round(float(predict_image_cifake(image_path)), decimals=2)
   percentage_predicted_t = (p1*w1+p2*w2)/(w1+w2)
   percentage_predicted_t = numpy.round((percentage_predicted_t),decimals=2)
   os.remove(image_path)
   return render_template('image/result_image_instagram.html', percentage_predicted_1=p1, percentage_predicted_2=p2, percentage_predicted_t=percentage_predicted_t)


@app.route('/result_video_manual', methods = ['GET', 'POST'])
def result_video_manual():
   file = request.files['file']
   video_path = download_video_manual(file)
   w1=94.44
   w2=99.0
   p1=predict_video_icpr(video_path)
   p2=predict_video_faceforensics(video_path)
   percentage_predicted_t = (p1*w1+p2*w2)/(w1+w2)
   percentage_predicted_t = numpy.round((percentage_predicted_t),decimals=2)
   os.remove(video_path)
   os.remove("./tmp/result.txt")
   return render_template('video/result_video_manual.html', percentage_predicted_1=p1, percentage_predicted_2=p2, percentage_predicted_t=percentage_predicted_t)


@app.route('/result_video_youtube', methods = ['GET', 'POST'])
def result_video_youtube():
   video_link = str(request.form.get('youtube_link', ''))
   video_path = get_video_youtube(video_link)
   w1=99.44
   w2=99.0
   p1=predict_video_icpr(video_path)
   p2=predict_video_faceforensics(video_path)
   percentage_predicted_t = (p1*w1+p2*w2)/(w1+w2)
   percentage_predicted_t = numpy.round((percentage_predicted_t),decimals=2)
   os.remove(video_path)
   os.remove("./tmp/result.txt")
   return render_template('video/result_video_youtube.html', percentage_predicted_1=p1, percentage_predicted_2=p2, percentage_predicted_t=percentage_predicted_t)



@app.route('/result_audio_manual', methods = ['GET', 'POST'])
def result_audio_manual():
   file = request.files['file']
   audio_path = download_audio_manual(file)
   p=predict_audio_vocoder(audio_path)
   percentage_predicted_t = p
   os.remove(audio_path)
   os.remove("./tmp/result_audio.txt")
   return render_template('audio/result_audio_manual.html', percentage_predicted_1=p, percentage_predicted_t=percentage_predicted_t)


@app.route('/result_audio_youtube', methods = ['GET', 'POST'])
def result_audio_youtube():
   audio_link = str(request.form.get('youtube_link', ''))
   audio_path = get_audio_youtube(audio_link)
   p=predict_audio_vocoder(audio_path)
   percentage_predicted_t = p
   os.remove(audio_path)
   os.remove("./tmp/result_audio.txt")
   return render_template('audio/result_audio_youtube.html', percentage_predicted_1=p, percentage_predicted_t=percentage_predicted_t)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html'), 500


@app.errorhandler(Exception)
def handle_exception(e):
    return render_template('error.html'), 500




def download_image_manual(file):
   rand = random.randint(100000, 999999)
   folder_path = "./tmp/"
   if not os.path.exists(folder_path):
      os.makedirs(folder_path)
   image_name = "image_manual_"+str(rand)+".jpg"
   path=folder_path+image_name
   file.save(path)
   return path


def download_video_manual(file):
   rand = random.randint(100000, 999999)
   folder_path = "./tmp/"
   if not os.path.exists(folder_path):
      os.makedirs(folder_path)
   video_name = "video_manual_"+str(rand)+".mp4"
   path=folder_path+video_name
   file.save(path)
   return path


def download_audio_manual(file):
   rand = random.randint(100000, 999999)
   folder_path = "./tmp/"
   if not os.path.exists(folder_path):
      os.makedirs(folder_path)
   audio_name = "audio_manual_"+str(rand)+".mp3"
   path=folder_path+audio_name
   file.save(path)
   return path



if __name__ == '__main__':
   app.run(debug = True)