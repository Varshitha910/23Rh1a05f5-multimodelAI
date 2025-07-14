#video extraction
import cv2
from transformers import BlipProcessor,BlipForConditionalGeneration
from PIL import Image
Processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")#creating the model
video_path="sample.mp4"
cap=cv2.VideoCapture(video_path)
success,frame=cap.read()
if success:
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV))
    inputs=Processor(images=image,return_tensors="pt")
    output=model.generate(**inputs)
    caption=Processor.decode(output[0])
    print(caption)