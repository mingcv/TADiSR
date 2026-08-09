import os

import torch
import torch.utils.checkpoint
import torch.utils.data

from tadisr_pipelines import CogView4TextEncoderPipeline


def main():
    quality_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations,text'
    negative_prompt = 'blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'

    saved_path = "./weights/CogView4/saved_prompt_tokens_nocfg.pt"
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    text_encoder = CogView4TextEncoderPipeline()
    print("=========== Start preparing the text embedding ===========")
    with torch.no_grad():
        text_encoder.prepare_text_embed(quality_prompt, negative_prompt, save_path=saved_path)
    print("=========== End preparing the text embedding ===========")


if __name__ == "__main__":
    main()
