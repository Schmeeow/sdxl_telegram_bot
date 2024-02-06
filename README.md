# stable_diffusion_telegram_bot (SDXL)

Простой телеграм-бот для генерации изображений с помощью AI-моделей Stable Diffusion (SDXL)

Бот генерирует картинку размером 1024x1024 в ответ на введенный пользователем текст (промт). 
Промты на английском обрабатываются напрямую, для перевода промтов с русского на английский используется нейросетевая модель машинного перевода Opus MT

Для генерации используется GPU (Cuda). При первом запуске будут автоматически загружены модели с hugginface (несколько десятков Гб)
 
Бот генерирует картинки в разных стилях, заданных используемой моделью и дополнениями к промту, переключая стиль генерации по командам
/basic, /photo, /art, /3d, /logo 

Можно сгенерировать сразу серию изображений по одному промту, указав в конце промта количество повторений после символов :: (например "кошка::10"). Во время генерации серии можно переключать стиль с помощью соответствующих команд.

Завершить серию после текущей картинки можно командой /stop

###################################################################

A simple telegram bot for generating images using Stable Diffusion (SDXL) AI models

The bot generates a 1024x 1024 image in response to the text entered by the user (prompt).
Prompts in English are processed directly, and the neural network machine translation model Opus MT is used to translate prompts from Russian to English

GPU (Cuda) is used for generation. At the first launch, models from hugginface.ai (~25 GB) will be automatically downloaded

The bot generates images in different styles specified by the model used and the prompt expansions, switching the generation style using the commands
/basic, /photo, /art, /3d, /logo

You can generate a series of images for the single prompt by specifying at the end of the prompt the number of repetitions after the characters :: (for example, "cat::10"). During the generation of a series, you can switch the style using the appropriate commands.

You can end the series after the current image with the /stop command

######################################################################

Список необходимых модулей приведен в requirements.txt
