import os
import re
import uuid
import pickle
import telebot
import time
import datetime
import requests
import torch
import numpy as np
from RealESRGAN import RealESRGAN
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline
from diffusers.utils import load_image, make_image_grid
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from PIL import Image
from io import BytesIO
from skimage import io
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from telebot import types

###################  ТОКЕН БОТА И РАЗРЕШЕНИЯ ####################

BOT_TOKEN = " "

###################  СТИЛИ ГЕНЕРАЦИИ ####################

COMMON_NEGATIVE_PROMPT = 'duplicate, blurry, disfigured, deformed, poorly drawn, extra limbs, watermark, long neck, elongated body, cropped image, deformed hands, twisted fingers, double image, malformed hands, multiple heads, ugly, poorly drawn hands, missing limb, lousy anatomy, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, out of focus, bad anatomy, disgusting, extra fingers, gross proportions, missing arms, cloned face, missing legs, tiling, poorly drawn feet, out of frame, blurred, signature, cut off, mutation,painting, mutated hands, bad proportions, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime, '

STYLES = {'basic': { 'model':'stablediffusionapi/juggernaut-xl-v7',
                     'type': 'sdxl',
                     'lora': {
                              'name':'',
                              'weights':'',
                              'activation_prompt':''
                              },
                     'pre_prompt':'',
                     'positive_expansion':'vibrant colors, volumetric light, cinematic, dynamic, elegant, sharp, vivid, pretty, aesthetic, fair quality, 4k',
                     'negative_expansion':'photo, art, cartoon, 3d',
                     'remix_prompt':'vibrant colors, volumetric light, cinematic, dynamic, elegant, sharp, vivid, pretty, aesthetic, fair quality, 4k'
                   },
          'photo': { 'model':'n0madic/colossusProjectXL_v53',
                     'type': 'sdxl',
                     'lora': {
                              'name':'ntc-ai/SDXL-LoRA-slider.nice-hands',
                              'weights':'nice hands.safetensors',
                              'activation_prompt':'nice '
                              },
                     'pre_prompt':'photo of ',
                     'positive_expansion':'photographic, realistic, realism, ((photography)), f/2.8, 35mm, highly detailed, intricate, cinematic, bokeh, sharp, raw photo, photorealistic, 4K, stock photo, natural soft light, fill light, detailed face, (film grain), vignette, DoF, pretty, aestetic',
                     'negative_expansion':'art, drawing, painting, illustration, anime, cartoon, graphic, text,  crayon, graphite, abstract, glitch, impressionist, noisy, blurry',
                     'remix_prompt':'((raw photo)), photographic, realistic, realism, ((photography)), natural, soft light, highly detailed, elegant, sharp, fine detail, fair quality,intricate, 4k'
                   },
          '3d':    { 'model': 'stablediffusionapi/protovisionxl-v3',
                     'type': 'sdxl',
                     'lora': {
                              'name':'goofyai/3d_render_style_xl',
                              'weights':'3d_render_style_xl.safetensors',
                              'activation_prompt': '3d, '
                              },
                     'pre_prompt':'3d render of ',
                     'positive_expansion':'3d model, video game character, volumetric light, 4k, 3d render, active, dynamic, highly detailed, 3d syle, dramatic light, elegant, sharp, vivid colors, fine detail, aesthetic, pretty, attractive, enhanced, bright, clear, CGI' ,
                     'negative_expansion':'photo, realism, art, painting, drawing, stock photo, photographic, realistic, 35mm film,illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, glitch, impressionist, noisy, blurry',
                     'remix_prompt':'((3d)), 3d model, vibrant, highly detailed, elegant, sharp, vivid, fine detail, fair quality, 4k'
                   },
          'art':   { 'model':'stablediffusionapi/artium-v20', 
                     'type': 'sdxl',
                     'lora': {
                              'name':'',
                              'weights':'',
                              'activation_prompt':''
                              },
                     'pre_prompt': 'artistic drawing of ',
                     'positive_expansion':'concept art in the style of {watercolor drawing | line art | pencil drawing | crayon art | pastel art | wet paint | digital painting | anime | Storybook Illustration | colored pencil art | Ballpoint Pen Art }, sketch, digital artwork, illustrative, painterly, matte painting, highly detailed',
                     'negative_expansion':'photo, 3d, cinematic, photography, realism, low contrast, stock photo, photograph, photographic, realistic, 35mm film, dslr, signature, watermark',
                     'remix_prompt':'((concept art)), ((drawing)), artistic, vibrant, high contrast, watercolor drawing, highly detailed, elegant, sharp, vivid, fine detail, fair quality, intricate, 4k'
                   },
          'logo':  { 'model':'jzli/realcartoonXL-v6',
                     'type': 'sdxl',
                     'lora': {
                              'name':'artificialguybr/LogoRedmond-LogoLoraForSDXL-V2',
                              'weights':'LogoRedmondV2-Logo-LogoRedmAF.safetensors',
                              'activation_prompt':'LogoRedAF'
                              },
                     'pre_prompt':'logotype of ',
                     'positive_expansion':'((logo)), schematic, sketch, mockup, design, vector art, (logotype), sign, symbol, original, unique, minimalistic, vector, flat, clean, (isolated on white background), contour, silhouette, simple, modern, color grading, high contrast',
                     'negative_expansion':'photo, realistic, realism, art, cartoon, 3d',
                     'remix_prompt':'((logo)), (isolated on white background), contour, silhouette, schematic, sketch, mockup, design, vector art, (logotype), sign, symbol, flat, line art',
                   },
          'nude':  { 'model':'stablediffusionapi/better-than-words',
                     'type': 'sdxl',
                     'lora': {
                              'name':'TonariNoTaku/SDXL_sufficient_nudity',
                              'weights':'nudity_v03XL_i1762_prod256n128b2_swn2_offset_e5.safetensors',
                              'activation_prompt':'1girl'
                              },
                     'pre_prompt':'naked photo of ',
                     'positive_expansion':'',
                     'negative_expansion':'',
                     'remix_prompt':'((nude)), ((naked)), nsfw'
                    }
}

###################  МЕНЮ БОТА ####################

SETTINGS_MENU = {'/style':  {  'select_text':'Cтиль для генерации/обработки изображений:',
                               'buttons': {'basic': 'Базовый', 'art':'Художественный', 'photo':'Фотореалистичный', '3d':'3D-графика', 'logo':'Логотип', 'nude':'18+'},
                               'reply_text':'Установлен стиль: '
                               },
                 '/ratio':  {  'select_text':'Cоотношение сторон для картинок:',
                               'buttons': {'1024,1024':'Квадрат (1:1)','768,1024':'Портрет (3:4)','1024,768':'Альбом (4:3)'},
                               'reply_text':'Установлено соотношение сторон: '
                                },
                 '/steps':  {  'select_text':'Качество генерации (лучше-дольше):',
                               'buttons':  {'20': 'Обычное', '40':'Высокое','60':'Супер'},
                               'reply_text':'Установлено качество: '
                               },
                 '/bg':     {  'select_text':'Цвет фона для режима изоляции:',
                               'buttons':  {'transparent':'Прозрачный', 'white': 'Белый', 'black':'Черный'},
                               'reply_text':'Установлен цвет фона: '
                                }
                }

###################  ПОЬЗОВАТЕЛИ ####################

USERS_DB_NAME = 'users.db'

users = {}

def add_user(user_id):
       global users
       users[user_id] = {}
       users[user_id]['mode'] = 'txt2img'
       users[user_id]['style'] = 'basic'
       users[user_id]['ratio'] = '1024,1024'
       users[user_id]['scale'] = 'fast'
       users[user_id]['steps'] = '20'
       users[user_id]['bg'] = 'transparent'

def save_users():
       global users
       with open('users.db', 'wb') as f:
         pickle.dump(users, f)
       print(len(users),"user(s) saved")

def load_users():
       global users
       file = USERS_DB_NAME
       try:
            with open (file, 'rb') as f:
                 users = pickle.load(f)
            print(len(users), "user(s) loaded")
       except IOError:
            pass

###################  ПЕРЕВОД ####################

def translate(prompt):
    chars = set('абвгдеёжзиклмнопрстуфхцшщэюяъь')
    if any((c in chars) for c in prompt):
         model_name = 'Helsinki-NLP/opus-mt-ru-en'
         tokenizer = AutoTokenizer.from_pretrained(model_name)
         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
         input_ids = tokenizer.encode(prompt, return_tensors="pt")
         output_ids = model.generate(input_ids, max_new_tokens=100)
         prompt = tokenizer.decode(output_ids[0], skip_special_tokens=True)
         print("Запрос на русском! Перевод: " + prompt + "\n")
    else:
       pass
    return(prompt)

###################  НАСТРОЙКИ ИНФЕРЕНСА ####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # works only on  nVIdia Ampere and later

###################  ИНФЕРЕНС ####################

def sysinfo(user_id, prompt, style, positive_prompt_final, negative_prompt_final):
    print("USER ID:", user_id)
    print("STYLE:", style)
    print("MODEL:", STYLES[style]['model'])
    print("LORA:", STYLES[style]['lora']['name'])
    print("POSITIVE:", positive_prompt_final)
    print("NEGATIVE:", negative_prompt_final)

def txt2img(user_id, prompt, style, ratio, steps):
    size = ratio.split(',')
    positive_prompt_final = STYLES[style]['pre_prompt'] + prompt + ", " + STYLES[style]['lora']['activation_prompt'] + ", " + STYLES[style]['positive_expansion']
    negative_prompt_final = COMMON_NEGATIVE_PROMPT + STYLES[style]['negative_expansion']
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        STYLES[style]['model'],
        torch_dtype=torch.float16,
    ).to(device)
    if not STYLES[style]['lora']['name'] == '':
        pipeline.load_lora_weights(STYLES[style]['lora']['name'], weight_name=STYLES[style]['lora']['weights'])
    sysinfo(user_id, prompt, style, positive_prompt_final, negative_prompt_final)
    pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipeline.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    image = pipeline(
            prompt = positive_prompt_final,
            negative_prompt = negative_prompt_final,
            num_inference_steps = int(steps),
            use_karras_sigmas=True,
            torch_dtype=torch.float16,
            cross_attention_kwargs={"scale": 0.9},
            original_size = (1024,1024),
            height = int(size[1]), width = int(size[0])
    ).images[0]
    image_path = "txt2image_" + user_id + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".jpg"
    image.save(image_path)
    print("filename: ", image_path + "\n")
    return(image_path)

def img2img(user_id, image_path, style, ratio, steps):    
    size = ratio.split(',')
    positive_prompt_final = STYLES[style]['remix_prompt'] + ", " + STYLES[style]['lora']['activation_prompt']
    negative_prompt_final = COMMON_NEGATIVE_PROMPT + STYLES[style]['negative_expansion']
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        STYLES[style]['model'],
        torch_dtype=torch.float16,
    ).to(device)
    if not STYLES[style]['lora']['name'] == '':
        pipeline.load_lora_weights(STYLES[style]['lora']['name'], weight_name = STYLES[style]['lora']['weights'])
    sysinfo(user_id=user_id, prompt='', style=style, positive_prompt_final=positive_prompt_final, negative_prompt_final=negative_prompt_final)
    init_image = load_image(image_path)
    pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipeline.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    image = pipeline(
            image = init_image,
            prompt = positive_prompt_final,
            negative_prompt = negative_prompt_final,
            strength = 0.7,
            num_inference_steps = int(steps),
            torch_dtype=torch.float16,
            cross_attention_kwargs={"scale": 0.9},
            use_karras_sigmas = True,
            original_size = (1024,1024),
            height = int(size[1]), width = int(size[0])
    ).images[0]
    make_image_grid([init_image, image], rows=1, cols=2)
    new_image_path = "img2img_" + user_id + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".jpg"
    image.save(new_image_path)
    return(new_image_path)

def img2iso(user_id, image_path, bg):
    net = BriaRMBG()
    net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
    net.to(device)
    net.eval()
    model_input_size = [1024,1024]
    orig_im = io.imread(image_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)
    result=net(image)
    result_image = postprocess_image(result[0][0], orig_im_size)
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
    orig_image = Image.open(image_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    if bg =='white' or bg =='black':
       image_extension = '.jpg'
       no_bg_image.load()
       if bg =='white':
          no_bg_image2 = Image.new("RGB", no_bg_image.size, (255, 255, 255))
       else:
          no_bg_image2 = Image.new("RGB", no_bg_image.size, (0, 0, 0))
       no_bg_image2.paste(no_bg_image, mask=no_bg_image.split()[3])
       no_bg_image = no_bg_image2
    else:
       image_extension = '.png'
    new_image_path = "img2iso_" + user_id + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + image_extension
    no_bg_image.save(new_image_path)
    return(new_image_path)

def img2ups(user_id, image_path):
    model = RealESRGAN(device, scale=2)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    image = Image.open(image_path).convert('RGB')
    upscaled_image = model.predict(image)
    upscaled_image.save(image_path)
    return(image_path)

###################  ФУНКЦИИ БОТА ####################

bot = telebot.TeleBot(BOT_TOKEN)
load_users()
style = 'basic'
need_to_stop = False

@bot.message_handler(commands = ['start'])
def start_message(message):
    bot.send_message(message.chat.id,"Привет! Пиши промт или присылай картинку. Стиль можно переключить в меню бота. Количество повторений (XX) можно указать, дописав ::XX в конце промта.")
    add_user(message.chat.id)
    save_users()
    print(users)

@bot.message_handler(commands = ['settings'])
def all_settings(message):
    #bot.send_message(message.chat.id, "⚙️ Настройки бота")
    keyboard_parts = dict()
    buttons_line = []
    settings = SETTINGS_MENU.keys()
    for setting in settings:
        keyboard_parts[setting] = telebot.types.InlineKeyboardMarkup()
        for key, value in SETTINGS_MENU[setting]['buttons'].items():
          if key == users[message.chat.id][setting[1:]]:
             keyboard_parts[setting].add(telebot.types.InlineKeyboardButton(text = "» " + value + " «", callback_data = setting + "+" + key))
          else:
             keyboard_parts[setting].add(telebot.types.InlineKeyboardButton(text = value, callback_data = setting + "+" + key))
        bot.send_message(message.chat.id, '⚙️ ' + SETTINGS_MENU[setting]['select_text'], reply_markup=keyboard_parts[setting])

@bot.callback_query_handler(func=lambda call:True)
def apply_settings(call):
    global style
    global users
    chat_id = call.message.chat.id
    split = call.data.split('+')
    settings_key = split[0][1:]
    settings_value =  split[1]
    message_id = call.message.id
    users[call.message.chat.id][settings_key] = str(settings_value)
    reply = SETTINGS_MENU[split[0]]['reply_text'] +  SETTINGS_MENU[split[0]]['buttons'][settings_value]
    bot.send_message(chat_id = chat_id, text = reply)
    save_users()

@bot.message_handler(commands = ['stop','isolate','upscale','create'])
def command_processor(message):
    global style
    global users
    global need_to_stop
    if message.text == '/isolate':
       users[message.chat.id]['mode'] = 'img2iso'
       bot.send_message(message.chat.id, "✂️ Включен режим изоляции. Цвет фона: " + SETTINGS_MENU['/bg']['buttons'][str(users[message.chat.id]['bg'])])
    elif message.text == '/upscale':
       users[message.chat.id]['mode'] = 'img2ups'
       bot.send_message(message.chat.id, "🔎 Включен режим увеличения")
    elif message.text == '/create':
       users[message.chat.id]['mode'] = 'txt2img'
       bot.send_message(message.chat.id, "🖼️ Включен режим генерации. Стиль: " + SETTINGS_MENU['/style']['buttons'][str(users[message.chat.id]['style'])])
    elif message.text == "/stop":
       need_to_stop = True
       bot.send_message(message.chat.id,"⛔Генерация остановится по завершении создания текущей картинки")
    save_users()

@bot.message_handler(content_types = ["text"])
def text_processor(message):
    global users
    m = re.search(r'(?<=::)\w+', message.text)
    if m is None:
       repeats = 1
    else:
       repeats = int(m.group(0))
    message.text = message.text.split('::', 1)[0]
    prompt = translate(message.text)
    for i in range(repeats):
        bot.send_message(message.chat.id, "⌛ Генерирую картинку (" + str(i+1) + "/" + str(repeats) + ") в стиле «" + SETTINGS_MENU['/style']['buttons'][str(users[message.chat.id]['style'])] + "», подождите...")
        generated_file_name = txt2img(str(message.chat.id), prompt, users[message.chat.id]['style'], users[message.chat.id]['ratio'], users[message.chat.id]['steps'])
        with open(generated_file_name, 'rb') as f:
              contents = f.read()
        bot.send_photo(message.chat.id, contents)
        os.remove(generated_file_name)

@bot.message_handler(content_types=['photo'])
def photo_processor(message):
    global users
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    save_path = 'temp_' + users[message.chat.id]['mode'] + "_" + uuid.uuid4().hex + '.jpg'
    with open(save_path, 'wb') as f:
        f.write(downloaded_file)
    if users[message.chat.id]['mode'] == 'img2img' or users[message.chat.id]['mode'] == 'txt2img':
       users[message.chat.id]['mode'] == 'img2img'
       bot.send_message(message.chat.id, "⌛ Обрабатываю картинку в стиле «" + SETTINGS_MENU['/style']['buttons'][str(users[message.chat.id]['style'])] + "», подождите...")
       generated_file_name = img2img(str(message.chat.id), save_path, users[message.chat.id]['style'], users[message.chat.id]['ratio'], users[message.chat.id]['steps'])
       with open(generated_file_name, 'rb') as f:
            contents = f.read()
       bot.send_photo(message.chat.id, contents)
    elif users[message.chat.id]['mode'] == 'img2iso':
       bot.send_message(message.chat.id, "⌛ Вырезаю фон, подождите...")
       generated_file_name = img2iso(str(message.chat.id), save_path, users[message.chat.id]['bg'])
       with open(generated_file_name, 'rb') as f:
            contents = f.read()
       bot.send_document(message.chat.id, document=contents, visible_file_name=generated_file_name)
    elif users[message.chat.id]['mode'] == 'img2ups':
       bot.send_message(message.chat.id, "⌛ Увеличиваю картинку, подождите...")
       generated_file_name = img2ups(str(message.chat.id), save_path)
       with open(generated_file_name, 'rb') as f:
           contents = f.read()
       bot.send_document(message.chat.id, document=contents, visible_file_name=generated_file_name)
    os.remove(save_path)
    os.remove(generated_file_name)

bot.infinity_polling()
