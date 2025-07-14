from flask import Flask,render_template,request#used to integrate html file
from transformers import BlipProcessor,BlipForConditionalGeneration
from PIL import Image
import os
processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
app=Flask(__name__)
UPLOAD_FOLDER="static/uploads"
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/imgtotext",methods=["POST","GET"])
def imgtotext():
    caption=""
    if request.method=="POST":
        file=request.files["image"]
        image_path=os.path.join(app.config["UPLOAD_FOLDER"],file.filename)
        file.save(image_path)
        if file:
            image=Image.open(image_path)
            inputs=processor(images=image,return_tensors="pt")
            output=model.generate(**inputs)
            caption=processor.decode(output[0])
           

    return render_template("imgtotext.html",caption=caption)
@app.route("/vidtotext",methods=["POST","GET"])
def vidtotext():
    return render_template("vidtotext.html")
if __name__=="__main__":
    app.run(debug=True)