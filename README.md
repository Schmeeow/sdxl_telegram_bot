Simple Diffusion Telegram Bot

Простой телеграм-бот для AI-генерации изображений по английским и русским промтам (перевод Opus MT) на моделях Stable Diffusion (SDXL), вырезания фона на картинках (Bria RMBG) и апскейлинга (Real-ESRGAN)

Бот генерирует картинки в разных стилях, определяемых используемой базовой моделью, LORA и дополнениями к промту (задается для каждого стиля в словаре конфигурации)
В меню бота настраивается стиль и качество генерации, соотношение сторон для картинки, цвет фона для изоляции. 
Для генерации используется GPU (Cuda). При первом запуске будут автоматически загружены модели с Hugging Face (несколько десятков Гб). Генерация одного изображения занимает около 30 секунд на GeForce 4060Ti(16Gb)
Можно сгенерировать сразу серию изображений по одному промту, указав в конце промта количество повторений после символов :: (например "кошка::10"). 
Во время генерации можно переключать стиль, соотношение сторон и качество с помощью соответствующих команд в настройках.

###################################################################

A simple telegram bot for AI image generation with english and russian prompts (Opus MT translation) on Stable Diffusion (SDXL) models, background removal (Bria RMBG) and upscaling (Real-ESRGAN)

The bot generates images in different styles, determined by the base model used, LORA and additions to the prompts (must be set for each style in the configuration dictionary)
In the bot menu, the style and quality of generation, the aspect ratio for the image, and the background color for isolated image are configured.
GPU (Cuda) is used for generation. At the first launch, models with Hugging Face (several tens of GB) will be automatically loaded. Generating a single image takes about 30 seconds on a GeForce 4060Ti(16Gb)
You can generate a series of images at once by specifying at the end of the prompt the number of repetitions after the characters :: (for example, "cat::10").
During generation, you can switch the style, aspect ratio and quality using the appropriate commands in the settings.

######################################################################

LINKS

https://github.com/Helsinki-NLP/Opus-MT

https://github.com/chenxwh/cog-RMBG

https://github.com/xinntao/Real-ESRGAN


TODO Список необходимых модулей приведен в requirements.txt

TODO Процесс установки описан в installation.txt


Развернутый бот - https://t.me/simplediffusiontelegrambot (может быть недоступен)
