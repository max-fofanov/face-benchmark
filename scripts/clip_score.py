import torch
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


image = Image.open("/path/to/test/image")
image_input = preprocess(image).unsqueeze(0).to(device)

text = "target prompt"
text_input = clip.tokenize([text]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)


image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (image_features @ text_features.T).cpu().numpy()
print(f"Similarity: {similarity}")
