from transformers import BlipProcessor,BlipForConditionalGeneration #this package or library file is uded for image to text
from PIL import Image#PIl=pillow
import requests#fetch the url from internet
import random
import torch#getting the value of transformers
Processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")#creating the processor
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")#creating the model
#to get any url
url="https://images.unsplash.com/photo-1747561246680-a34121708d80"
image=Image.open(requests.get(url,stream=True).raw).convert("RGB")
#create and initialize model and fetch text
inputs=Processor(image,return_tensors="pt")#to convert into pixels and giving the input to the processor,reading the image
output=model.generate(**inputs)
caption=Processor.decode(output[0])
#creating hashtags for captions
hashtags=["#flowers","#beautiful","#nature","#rose"]#hashtags for images
print(caption)
print(random.choice(hashtags))
#combing the hastags and image to print as chart
chart_caption=f"{caption}.{random.choice(hashtags)}"
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis("off")
plt.title(chart_caption,fontsize=10)
plt.show()