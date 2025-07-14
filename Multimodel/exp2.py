from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests
url="https://images.unsplash.com/photo-1747561246680-a34121708d80"
image=Image.open(requests.get(url,stream=True).raw).convert("RGB")
#create and process model
processor=BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").eval()
#Ask question about image
question="which flower is in the image?"
inputs=processor(image,question,return_tensors="pt")
output=model.generate(**inputs)
answer=processor.decode(output[0],skip_special_tokens=True)
print(question)
print(answer)