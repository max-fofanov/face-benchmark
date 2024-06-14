from safetensors.torch import load_file
from diffusers import AutoPipelineForText2Image
import torch


file = "/path/to/textual_inversion"
state_dict = load_file(file)

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_textual_inversion(state_dict["clip_g"], token="<token>", text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
pipe.load_textual_inversion(state_dict["clip_l"], token="<token>", text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

result = pipe("<token>").images[0]
