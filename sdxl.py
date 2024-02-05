from diffusers import AutoPipelineForText2Image
import torch
import datetime

COMMON_NEGATIVE_PROMPT = 'duplicate, blurry, disfigured, deformed, poorly drawn, extra limbs, watermark, blur haze, 2 heads, long neck, elongated body, cropped image, out of frame, deformed hands, twisted fingers, double image, malformed hands, multiple heads, extra limb, ugly, poorly drawn hands, missing limb, cut-off, over statured, lousy anatomy, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, out of focus, long body, disgusting, extra fingers, gross proportions, missing arms, mutated hands, cloned face, missing legs, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, blurred, grainy, signature, cut off, lousy anatomy, disfigured, poorly drawn face, mutation, '

STYLES = { 'basic':{ 'model':'jzli/realcartoonXL-v6',
                     'positive_expansion':'vibrant, highly detailed, intricate, elegant, sharp, vivid, fine detail, fair quality',
                     'negative_expansion':'photo, art, cartoon, 3d'
                   },
          'photo': { 'model':'n0madic/colossusProjectXL_v53',
                     'positive_expansion':'photographic, realistic, realism, photography, Zeiss, f/2.8, 35mm photo, highly detailed, cinematic, bokeh, intricate, elegant, sharp, professional, fine detail, aesthetic, pretty, raw photo, photorealistic, HDR, 4K, stock photo, photograph, natural soft light, detailed face',
                     'negative_expansion':'art, drawing, painting, drawing, illustration, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, impressionist, noisy, blurry'
                   },
          '3d':    { 'model':'Yntec/3DRendering',
                     'positive_expansion':'3d model, videogame, vibrant rim light, volumetric light, 4k, active, dynamic, highly detailed, cinematic, intricate, dramatic light, elegant, sharp, vivid, fine detail, fair quality, aesthetic, pretty, attractive, enhanced, color, bright, artistic, amazing, symmetry, clear, artistic, CGI',
                     'negative_expansion':'photo, realism, art, painting, drawing, stock photo, photographic, realistic, realism, 35mm film, dslr,, art, drawing, painting, drawing, illustration, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, impressionist, noisy, blurry'
                   },
          'art':   { 'model':'jzli/realcartoonXL-v6',
                     'positive_expansion':'concept art in the style of {watercolor drawing | line art | blue print | pencil drawing | crayon art | pastel art | wet paint | digital painting | anime | Storybook Illustration | colored pencil art | Ballpoint Pen Art}, digital artwork, illustrative, painterly, matte painting, highly detailed',
                     'negative_expansion':'photo, 3d, cinematic, photography, realism, low contrast, stock photo, photograph, photographic, realistic, 35mm film, dslr'
                   },
}

def txt2img(user_id, prompt, style):

    prompt = prompt + ", "

    print("user id: ", user_id, "\n")
    print("style: ", style, "\n")
    print("model: ", STYLES[style]['model'], "\n")
    print("prompt: ", prompt, STYLES[style]['positive_expansion'], "\n")
    print("negative prompt: ", COMMON_NEGATIVE_PROMPT, STYLES[style]['negative_expansion'], "\n")

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        STYLES[style]['model'],
        torch_dtype=torch.float16,
        #variant='fp16',
        use_safetensors=True
    ).to("cuda")

    image = pipeline_text2image(
            prompt = prompt + STYLES[style]['positive_expansion'],
            negative_prompt = COMMON_NEGATIVE_PROMPT + STYLES[style]['negative_expansion'],
            height = 1024,
            width = 1024
    ).images[0]

    image_path = "image_" + user_id + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".jpg"
    image.save(image_path)

    print("filename: ", image_path + "\n")

    return(image_path)

