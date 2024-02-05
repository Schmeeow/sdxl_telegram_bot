import telebot
import sdxl
import translate

BOT_TOKEN = "6942638415:AAGvG7fnNqq3bZUoVvogL6QgOTHrZXqRiiI"

bot=telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])

def ru_detector(s):
    chars = set('абвгдеёжзиклмнопрстуфхцшщэюяъь')
    if any((c in chars) for c in s):
       return True
    else:
       return False


def start_message(message):
  bot.send_message(message.chat.id,"Привет! пиши что надо нарисовать")


@bot.message_handler(content_types=["text"])

def repeat_all_messages(message):

    ## русский переводим английский пропускаем
    if ru_detector(message.text):
       english_text = translate.ru2en(message.text)
       #bot.send_message(message.chat.id, "Запрос на русском! Вот перевод твоего запроса на английcкий: " + english_text)
    else:
       english_text = message.text
       #bot.send_message(message.chat.id, "Запрос на английском: " + english_text)

    bot.send_message(message.chat.id, "Генерирую картинку, подожди 20 секунд...")

    style = "3d"

    with open(sdxl.txt2img(str(message.chat.id),english_text,style), 'rb') as f:
         contents = f.read()
    bot.send_photo(message.chat.id, contents)

    style = "photo"

    with open(sdxl.txt2img(str(message.chat.id),english_text,style), 'rb') as f:
         contents = f.read()
    bot.send_photo(message.chat.id, contents)

    style = "art"

    with open(sdxl.txt2img(str(message.chat.id),english_text,style), 'rb') as f:
       contents = f.read()
    bot.send_photo(message.chat.id, contents)

    style = "basic"

    with open(sdxl.txt2img(str(message.chat.id),english_text,style), 'rb') as f:
       contents = f.read()
    bot.send_photo(message.chat.id, contents)


bot.infinity_polling()
