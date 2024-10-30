# -*- coding: utf-8 -*-
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig
import torch
from transformers.generation.configuration_utils import GenerationConfig

from lmms_eval.models.Emu3.emu3.mllm.processing_emu3 import Emu3Processor

def i2t_generate(
    model, 
    tokenizer,
    image_processor,
    image_tokenizer,
    text,
    image, 
    generation_config
):
    torch.cuda.empty_cache()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    inputs = processor(
        text=text,
        image=image,
        mode='U',
        padding_image=True,
        padding="longest",
        return_tensors="pt",
    )

    # generate
    outputs = model.generate(
        inputs.input_ids.to(model.device),
        generation_config,
        attention_mask=inputs.attention_mask
    )

    outputs = outputs[:, inputs.input_ids.shape[-1]:]
    answers = processor.batch_decode(outputs, skip_special_tokens=True)
    
    return answers


if __name__ == '__main__':

    # EMU_HUB = "BAAI/Emu3-Chat"
    EMU_HUB = "lmms_eval/models/Emu3/Emu3-Chat-3bit"
    VQ_HUB = "BAAI/Emu3-VisionTokenizer"
    text = "Please describe the image"
    image = Image.open("lmms_eval/models/Emu3/assets/demo.png")
    
    tokenizer = AutoTokenizer.from_pretrained("BAAI/Emu3-Chat", 
                                              trust_remote_code=True, 
                                              padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, 
                                                         load_in_8bit=True,
                                                         trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:3", trust_remote_code=True).eval()
    
    # gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        EMU_HUB, 
        device_map="cuda:3",
        # torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        # quantization_config=gptq_config,
        # load_in_8bit=True,
    )
    
    model._supports_quantized_cache = True
    answers = i2t_generate(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_tokenizer=image_tokenizer,
        text=text,
        image=image,
        generation_config=GenerationConfig(
                  pad_token_id=tokenizer.pad_token_id,
                  bos_token_id=tokenizer.bos_token_id,
                  eos_token_id=tokenizer.eos_token_id,
                  max_new_tokens=10,
                  cache_implementation="quantized",
                  cache_config={"backend": "quanto", "nbits": 4}
        )
    )
    
    print(answers)
    