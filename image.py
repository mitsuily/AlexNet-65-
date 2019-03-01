from PIL import Image
import os.path
import glob

def convertjpg(jpgfile,outdir,width=32,height=32):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("/home/mitsui/AlexNet/dataset/train/33/*.jpg"):
    convertjpg(jpgfile,"/home/mitsui/AlexNet/dataset1/train/33/")