import telebot
import sdxl
import translate
import re

BOT_TOKEN = ""

bot=telebot.TeleBot(BOT_TOKEN)

style="basic"

@bot.message_handler(commands=['basic','art', '3d', 'photo', 'logo'])

def switch_style(message):
    global style
    if message.text == "/basic":
         style="basic"
    elif message.text == "/3d":
         style="3d"
    elif message.text == "/photo":
         style="photo"
    elif message.text == "/art":
         style="art"
    elif message.text == "/logo":
         style="logo"
    bot.send_message(message.chat.id,"Стиль изменен на " + style)

@bot.message_handler(commands=['start'])

def start_message(message):
    bot.send_message(message.chat.id,"Привет! Пиши промт. Стиль можно переключить в меню бота")


@bot.message_handler(content_types=["text"])
def get_a_prompt(message):

    # ищем в тексте :количество_повторений, если не найдено, повторяем один раз
    m = re.search(r'(?<=:)\w+', message.text)
    if m is None:
         repeats=1
    else:
         repeats=int(m.group(0))

    # русский промт переводим, английский пропускаем так
    if translate.ru_detector(message.text):
       english_text = translate.ru2en(message.text)
    else:
       english_text = message.text

    # генерируем картинку с нужным стилем и нужным количеством повторений
    for i in range(repeats):
          bot.send_message(message.chat.id, "Генерирую картинку №" + str(i+1) + " из " + str(repeats) + " в стиле " + style + ", подождите 20 секунд...")
          with open(sdxl.txt2img(str(message.chat.id),english_text,style), 'rb') as f:
               contents = f.read()
          bot.send_photo(message.chat.id, contents)

bot.infinity_polling()
