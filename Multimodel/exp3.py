#image to text then translate to regional language
from transformers import BlipProcessor, BlipForQuestionAnswering,BlipForConditionalGeneration
from PIL import Image
import requests
from transformers import MBartForConditionalGeneration,MBart50TokenizerFast
Processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")#creating the processor
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")#creating the model
#to get any url
url="https://images.unsplash.com/photo-1747561246680-a34121708d80"
image=Image.open(requests.get(url,stream=True).raw).convert("RGB")
#create and initialize model and fetch text
inputs=Processor(image,return_tensors="pt")#to convert into pixels and giving the input to the processor,reading the image
output=model.generate(**inputs)
caption=Processor.decode(output[0])
#load MBart model
model_name="facebook/mbart-large-50-many-to-many-mmt"
tokenizer=MBart50TokenizerFast.from_pretrained(model_name)
model=MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer.src_lang="en_XX"#present in which language
target_lang="te_IN"#transfer into which language
inp_lng=tokenizer(caption,return_tensors="pt")
out_tok=model.generate(**inp_lng,forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
caption_te=tokenizer.decode(out_tok[0])
print(caption_te)