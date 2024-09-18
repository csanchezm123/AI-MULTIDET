import re
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import pickle
import torch
import cv2
import numpy
from torch.utils.model_zoo import load_url
from scipy.special import expit
import sys
import os
import multiprocessing
sys.path.append('..')

from models.video.icpr2020dfdc.blazeface import FaceExtractor, BlazeFace, VideoReader
from models.video.icpr2020dfdc.architectures import fornet,weights
from models.video.icpr2020dfdc.isplutils import utils


stop_words = set(stopwords.words('english'))

'''PREDICCIÓN CON MODELO LLM'''

def predict_normal_text(text):
    join_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    words = join_text.split()  # Tokenize
    words = [word.lower() for word in words if word.isalpha()]  # Lowercase and remove non-alphabetic words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    clean_text = ' '.join(words)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding=True, truncation=True, max_length=128)
    token_text = tokenizer(clean_text, padding=True, truncation=True, return_tensors='pt')
    #Load model and get the prediction
    with open('./models/text/model_text_normal.pickle', "rb") as file:
        text_normal_model = pickle.load(file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_normal_model.to(device)
        text_normal_model.eval()
        token_text = token_text['input_ids'].to(device)
        with torch.no_grad():
            prediction = text_normal_model(token_text)
        logits = prediction.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction_probabilities = probabilities.squeeze().tolist()
        prediction_result = round((prediction_probabilities[1]*100),2)
    file.close()
    return prediction_result


'''PREDICCIÓN CON MODELO FAKE NEWS DETECTION'''

def predict_news_text(text):
    #Load model and tokenizer
    news_model = BertForSequenceClassification.from_pretrained('./models/text/trained_model_beto')
    tokenizer = BertTokenizer.from_pretrained('./models/text/trained_model_beto')
    news_model.eval()
    token_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    #Get the prediction
    with torch.no_grad():
        prediction = news_model(**token_text)
    logits = prediction.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction_probabilities = probabilities.squeeze().tolist()
    #Adjust the prediction
    prediction_result = round((prediction_probabilities[1]*100),2)
    return prediction_result


'''PREDICCIÓN PARA TEXTO PROVENIENTE DE X'''

def predict_text_x(text):
    #Load model and tokenizer
    news_model = RobertaForSequenceClassification.from_pretrained('./models/text/trained_model_tweepfake')
    tokenizer = RobertaTokenizer.from_pretrained('./models/text/trained_model_tweepfake')
    news_model.eval()
    token_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    #Get the prediction
    with torch.no_grad():
        prediction = news_model(**token_text)
    logits = prediction.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction_probabilities = probabilities.squeeze().tolist()
    #Adjust the prediction
    prediction_result = round((prediction_probabilities[1]*100),2)
    return prediction_result



'''PREDICCIÓN CON MODELO CIFAKE'''

def predict_image_cifake(image_path):
    target_size = (32, 32)
    batch_size = 32
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = numpy.expand_dims(image, axis=0)
    with open('./models/image/CIFAKE/cifake.pickle', 'rb') as file:
        model = pickle.load(file)
        prediction_human = model.predict(image)
    file.close()
    prediction = 1 - prediction_human
    prediction = numpy.round((prediction*100),decimals=2)
    return prediction


'''PREDICCIÓN CON MODELO ICPR2020DFDC'''

def predict_video_icpr(file):
    net_model = 'EfficientNetAutoAttB4'
    train_db = 'DFDC'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224
    frames_per_video = 32
    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))
    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    facedet = BlazeFace().to(device)
    facedet.load_weights("./models/video/icpr2020dfdc/blazeface/blazeface.pth")
    facedet.load_anchors("./models/video/icpr2020dfdc/blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)
    faces = face_extractor.process_video(file)
    faces_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in faces if len(frame['faces'])] )
    with torch.no_grad():
        faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()
        pred_mean = float(format(expit(faces_pred.mean())))
        pred_mean = numpy.round((pred_mean*100),decimals=2)
    return pred_mean


'''PREDICCIÓN CON MODELO FACEFORENSICS++'''


def run_script_with_args(script_name, *args):
    command = f"python {script_name} " + " ".join(args)
    print(command)
    with os.popen(command) as process:
        output = process.read()
    return output

def predict_video_faceforensics(file):
    queue = multiprocessing.Queue()
    script_name = "./models/video/FaceForensics/classification/detect_from_video.py"
    input = file
    output_path = "./tmp"
    args = ( "--video_path " + input, " --output_path " + output_path + " --cuda ")
    result = run_script_with_args(script_name, *args)
    with open(output_path+"/result.txt", 'r') as file:
        prediction = file.read()
        prediction = float(prediction)
    file.close()
    prediction = numpy.round(prediction*100, decimals=2)
    return prediction


'''PREDICCIÓN CON MODELO VOCODERS ARTIFACTS'''

def predict_audio_vocoder(file):
    queue = multiprocessing.Queue()
    script_name = "./predict_audio_vocoder.py"
    input = file
    model_path = "./models/audio/VocodersArtifacts/librifake_pretrained_lambda0.5_epoch_25.pth"
    output_path = "./tmp/"
    args = ( " --input_path " + input, " --model_path " + model_path + " --output_path " + output_path)
    result = run_script_with_args(script_name, *args)
    with open("./tmp/result_audio.txt", 'r') as file:
        prediction = file.read()
        prediction = float(prediction)
    file.close()
    prediction = numpy.round(prediction*100, decimals=2)
    return prediction