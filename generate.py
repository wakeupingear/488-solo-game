from math import ceil
import tensorflow_hub as hub
import tensorflow as tf
from simplejson import load
from urllib.parse import urlparse
from ast import arg
from PIL import ImageColor, Image, ImageDraw, ImageFont
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from fpdf import FPDF
import glob

import os
import sys
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Importing tensorflow...")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Loading model...")
hub_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
imageSize = 512

f = open("characters.json", "r+")
data = json.load(f)
borderColor = ImageColor.getcolor(data["borderColor"], "RGBA")
borderWidth = data["borderWidth"]
topPadding = data["topPadding"]

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = imageSize / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def load(url):
    path = url.replace("https://", "").replace("http://",
                                               "").replace("www.", "").replace("/", "")
    file = tf.keras.utils.get_file(
        path, url)
    return file


def wrap_text(text, width, font):
    text_lines = []
    text_line = []
    text = text.replace('\n', ' [br] ')
    words = text.split()

    for word in words:
        if word == '[br]':
            text_lines.append(' '.join(text_line))
            text_line = []
            continue
        text_line.append(word)
        w, h = font.getsize(' '.join(text_line))
        if w > width:
            text_line.pop()
            text_lines.append(' '.join(text_line))
            text_line = [word]

    if len(text_line) > 0:
        text_lines.append(' '.join(text_line))

    return text_lines


def drawText(image, msg, offset, fnt, fill, maxWidth=700):
    draw = ImageDraw.Draw(image)
    baseW, baseH = draw.textsize("KSASFLI", font=fnt)
    msg = wrap_text(msg, maxWidth, fnt)
    height = 0
    lineCount=0
    for i, line in enumerate(msg):
        w, h = draw.textsize(line, font=fnt)
        height = (baseH*1.4)*(i)
        lineCount+=1
        draw.text((-w//2+offset[0], baseH//2+offset[1] +
                  height), line, fill=fill, font=fnt)
    return (baseH*1.4)*(lineCount)+baseH*0.6

def drawImage(image, offset, card, colored):
    draw=-1
    if colored:
        draw = ImageDraw.Draw(card)
        draw.rectangle((offset[0]-borderWidth, offset[1], offset[0] +
                   imageSize+borderWidth, offset[1]+imageSize+borderWidth*2), borderColor)
    else:
        draw = ImageDraw.Draw(card)
        draw.rectangle((offset[0]-borderWidth, offset[1], offset[0] +
                   imageSize+borderWidth, offset[1]+imageSize+borderWidth*2), "#888888")
    card.paste(image, (int(offset[0]), int(offset[1]+borderWidth)))
    return image.size[1]+borderWidth*2


def main():
    defaultStyle = load(data["styles"][0])
    widthHalved = (data["width"] - imageSize)//2
    offset = (widthHalved, topPadding)

    bgColor = ImageColor.getcolor(data["bgColor"], "RGBA")
    frontColorTemplate = Image.new(
        "RGBA", (data["width"], data["height"]), bgColor)
    frontBlackTemplate = Image.new(
        "RGBA", (data["width"], data["height"]), "#FFFFFF")
    backColorTemplate = frontColorTemplate.copy()
    backBlackTemplate = frontBlackTemplate.copy()

    if not os.path.exists("art"):
        print("Creating new art directory")
        os.makedirs("art")
    if not os.path.exists("art/color"):
        print("Creating new color directory")
        os.makedirs("art/color")
    if not os.path.exists("art/black"):
        print("Creating new black directory")
        os.makedirs("art/black")

    # check first cli argument
    if len(sys.argv) > 1:
        key = sys.argv[1:]
        key = ' '.join(key)
        key = key.replace("[", "").replace("]", "")
        key = key.split(",")
        for character in key:
            character = character.strip()
            found = False
            for obj in data["characters"]:
                if (character == obj["name"]):
                    process(obj, data, defaultStyle,
                            frontColorTemplate, backColorTemplate, frontBlackTemplate, backBlackTemplate, offset)
                    found = True
                    break
            if not found:
                print("Character not found")
    else:
        for character in data["characters"]:
            process(character, data, defaultStyle,
                    frontColorTemplate, backColorTemplate, frontBlackTemplate, backBlackTemplate, offset)
    if not data["savePdf"]:
        exit(0)
    print("Creating color fronts...")
    createPDF("color", "front", False)
    print("Creating color backs...")
    createPDF("color", "back", True)

    print("Creating black fronts...")
    createPDF("black", "front", False)
    print("Creating black backs...")
    createPDF("black", "back", True)


def createPDF(cType, side, reversed):
    pdf = FPDF()
    pdf.add_page()
    cardW = 70
    cardH = 93
    x = 0
    y = 0
    print(cType)
    filelist = glob.glob("./art/"+cType+"/*.png")
    for image in sorted(filelist):
        if (side in image):
            pdf.image(image, x=(x*cardW if (not reversed)
                      else cardW*(2-x)), y=y*cardH, w=cardW, h=cardH)
            x += 1
            if (x == 3):
                x = 0
                y += 1
                if (y == 3):
                    y = 0
                    pdf.add_page()

    pdf.output(side+cType.capitalize()+"Template.pdf", "F")


def process(character, data, defaultStyle, frontColorTemplate, backColorTemplate, frontBlackTemplate, backBlackTemplate, offset):
    print(character["name"]+"...")
    image = character["image"].split("?")[0]
    content_path = load(image,)
    content_image = load_img(content_path)
    style_path = defaultStyle
    if ("style" in character):
        # check if the style is a string
        if (type(character["style"]) is str):
            style_path = load(character["style"])
        else:
            style_path = load(data["styles"][character["style"]])
    style_image = load_img(style_path)

    if (data["useStyle"]):
        stylized_image = hub_model(tf.constant(
            content_image), tf.constant(style_image))[0]
        stylized_image = tensor_to_image(stylized_image)
    else:
        stylized_image = tensor_to_image(content_image)

    width, height = stylized_image.size
    size = min(width, height)
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    cropOff = (0, 0)
    if ("crop" in character):
        cropOff = character["crop"]
    stylized_image = stylized_image.crop(
        (left+cropOff[0], top+cropOff[1], right+cropOff[0], bottom+cropOff[1]))
    stylized_image = stylized_image.resize((imageSize, imageSize))

    words = character["name"].split(" ")
    name = "".join(words[:min(len(words), 2)])
    writeImages(stylized_image, "./art/color/"+name, character, offset,
                frontColorTemplate, backColorTemplate)
    writeImages(stylized_image, "./art/black/"+name, character, offset,
                frontBlackTemplate, backBlackTemplate)

    return name


def writeImages(image, fileName, character, offset, frontTemplate, backTemplate):
    imageWidth, imageHeight = image.size
    color = "white" if ("color" in fileName) else "black"
    fnt = ImageFont.truetype("./FreeMono.ttf", 80)
    fntSmall = ImageFont.truetype("./FreeMono.ttf", 60)
    traits = "\n".join(character["traits"])

    front = frontTemplate.copy()
    height = offset[1]
    height+=drawText(front, character["name"],
             (front.size[0]//2, height), fnt, color)
    height+=100
    height+=drawImage(image, (offset[0],height), front, color=="white")
    front.save(fileName+"_front.png")

    
    back = backTemplate.copy()
    height = offset[1]
    height+=drawText(back, character["name"],
             (front.size[0]//2, height), fnt, color)
    height+=drawImage(image, (offset[0],height), back, color=="white")
    height+=drawText(back, traits, (back.size[0]//2, height), fntSmall, color)
    back.save(fileName+"_back.png")


if __name__ == "__main__":
    main()
