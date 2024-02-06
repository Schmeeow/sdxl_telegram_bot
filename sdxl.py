
from diffusers import AutoPipelineForText2Image
import torch
import datetime

# общий негативный промт для всех генераций
COMMON_NEGATIVE_PROMPT = 'duplicate, blurry, disfigured, deformed, poorly drawn, extra limbs, watermark, long neck, elongated body, cropped image, deformed hands, twisted fingers, double image, malformed hands, multiple heads, extra limb, ugly, poorly drawn hands, missing limb'

# описание стилей генерации
STYLES = {'basic': { 'model':'jzli/realcartoonXL-v6',
                     'pre_prompt':'',
                     'positive_expansion':'vibrant, highly detailed, intricate, elegant, sharp, vivid, fine detail, fair quality',
                     'negative_expansion':'photo, art, cartoon, 3d'
                   },
          'photo': { 'model':'n0madic/colossusProjectXL_v53',
                     'pre_prompt':'photo of ',
                     'positive_expansion':'photographic, realistic, realism, photography, f/2.8, 35mm photo, highly detailed, intricate, cinematic, bokeh, elegant, sharp, fine detail, aesthetic, pretty, raw photo, photorealistic, 4K, stock photo, natural soft light, detailed face',
                     'negative_expansion':'art, drawing, painting, drawing, illustration, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, impressionist, noisy, blurry'
                   },
          '3d':    { 'model': 'stablediffusionapi/protovisionxl-v3',
                     'pre_prompt':'3d render of ',
                     'positive_expansion':'3d model, video game character, volumetric light, 4k, 3d render, active, dynamic, highly detailed, dimensional, dramatic light, elegant, sharp, vivid colors, fine detail, fair quality, aesthetic, pretty, attractive, enhanced, bright, clear',
                     'negative_expansion':'photo, realism, art, painting, drawing, stock photo, photographic, realistic, realism, 35mm film, art, painting, drawing, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, glitch, impressionist, noisy, blurry'
                   },
          'art':   { 'model':'jzli/realcartoonXL-v6',
                     'pre_prompt': 'artistic drawing of ',
                     'positive_expansion':'concept art in the style of {watercolor drawing | line art | pencil drawing | crayon art | pastel art | wet paint | digital painting | anime | Storybook Illustration | colored pencil art | Ballpoint Pen Art }, sketch, digital artwork',
                     'negative_expansion':'photo, 3d, cinematic, photography, realism, low contrast, stock photo, photograph, photographic, realistic, 35mm film, dslr, signature, watermark'
                   },
          'logo':  { 'model':'stabilityai/stable-diffusion-xl-base-1.0',
                     'pre_prompt':'logotype of ',
                     'positive_expansion':'((logo)), schematic, sketch, mockup, design, vector art, (logotype), sign, symbol, original, unique, logo, logotype, minimalistic, vector, flat, clean, simple, modern, white background, color grading, high contrast',
                     'negative_expansion':'photo, realistic, realism, art, cartoon, 3d'
                   }
}

# возвращает имя сгенерированного файла (image_ID_ПОЛЬЗОВАТЕЛЯ_ВРЕМЯ_ДАТА.jpg)
def txt2img(user_id, prompt, style):

    positive_prompt_final = STYLES[style]['pre_prompt'] + prompt + ", " + STYLES[style]['positive_expansion']
    negative_prompt_final = COMMON_NEGATIVE_PROMPT + STYLES[style]['negative_expansion']

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        STYLES[style]['model'],
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    image = pipeline_text2image(
            prompt = positive_prompt_final,
            negative_prompt = negative_prompt_final,
            num_inference_steps = 55,
            use_karras_sigmas=True,
            original_size = (1024,1024),
            height = 1024,
            width = 1024
    ).images[0]

    image_path = "image_" + user_id + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".jpg"
    image.save(image_path)

    return(image_path)

