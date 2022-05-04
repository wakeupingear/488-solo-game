from asyncio.windows_events import NULL
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
from math import floor

import os
import sys

from torch import clamp_min
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Importing tensorflow...")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Loading model...")
hub_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
imageSize = 630

f = open("characters.json", "r+")
data = json.load(f)
borderWidth = data["borderWidth"]
topPadding = data["topPadding"]

borderColorDefault = data["borderColorDefault"]
borderImageBase = Image.open(data["borderImage"])
borderImageBase = borderImageBase.resize((data["width"], data["height"]))
border = {}
borderBackImageBase = Image.open(data["borderBackImage"])
borderBackImageBase = borderBackImageBase.resize(
    (data["width"], data["height"]))
borderBack = {}
print("Creating outlines...")
for key, value in data["borderColors"].items():
    color = value.lstrip("#")
    rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    imgData = np.array(borderImageBase)
    for x in range(imgData.shape[0]):
        for y in range(imgData.shape[1]):
            if imgData[x, y][3] > 0:
                imgData[x, y] = (max(rgb[0]+imgData[x,y][0]-255,0), max(rgb[1]+imgData[x,y][1]-255,0), max(rgb[2]+imgData[x,y][2]-255,0), imgData[x, y][3])
    border[key] = Image.fromarray(imgData)

    imgData = np.array(borderBackImageBase)
    for x in range(imgData.shape[0]):
        for y in range(imgData.shape[1]):
            if imgData[x, y][3] > 0:
                imgData[x, y] = (max(rgb[0]+imgData[x,y][0]-255,0), max(rgb[1]+imgData[x,y][1]-255,0), max(rgb[2]+imgData[x,y][2]-255,0), imgData[x, y][3])
    borderBack[key] = Image.fromarray(imgData)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    if (path_to_img.endswith(".png")):
        img = tf.image.decode_png(img, channels=4)
    else:
        img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = imageSize / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def load(url):
    file = NULL
    if os.path.exists(url):
        file = url
    else:
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


def drawText(image, msg, offset, fnt, fill, maxWidth=650, stroke_width=5):
    draw = ImageDraw.Draw(image)
    baseW, baseH = draw.textsize("KSASFLI", font=fnt)
    msg = wrap_text(msg, maxWidth, fnt)
    height = 0
    lineCount = 0
    for i, line in enumerate(msg):
        w, h = draw.textsize(line, font=fnt)
        height = (baseH*1.4)*(i)
        lineCount += 1
        draw.text((-w//2+offset[0], baseH//2+offset[1] +
                  height), line, fill=fill, font=fnt,
                  stroke_width=stroke_width,
                  stroke_fill="black")
    return (baseH*1.4)*(lineCount)+baseH*0.6


def drawImage(image, offset, card, colored):
    draw = -1
    if colored:
        draw = ImageDraw.Draw(card)
        #draw.rectangle((offset[0]-borderWidth, offset[1], offset[0] + imageSize+borderWidth, offset[1]+imageSize+borderWidth*2), borderColor)
    else:
        draw = ImageDraw.Draw(card)
        # draw.rectangle((offset[0]-borderWidth, offset[1], offset[0] + imageSize+borderWidth, offset[1]+imageSize+borderWidth*2), "#888888")
    card.paste(image, (int(offset[0]), int(offset[1]+borderWidth)), mask=image)
    return image.size[1]+borderWidth*2


def main():
    defaultStyle = load(data["styles"][0])
    widthHalved = (data["width"] - imageSize)//2
    offset = (widthHalved, topPadding)

    bgColor = ImageColor.getcolor(data["bgColor"], "RGBA")
    frontColorTemplate = Image.new(
        "RGBA", (data["width"], data["height"]), bgColor)
    if data["bgImage"] != "":
        bgImage = Image.open(data["bgImage"])
        bgImage = bgImage.resize((data["width"], data["height"]))
        frontColorTemplate.paste(bgImage, (0, 0))
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
        exit(0)
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
    cardW = 70
    cardH = 93
    x = 0
    y = 0
    count = 0
    filelist = glob.glob("./art/"+cType+"/*.png")
    for image in sorted(filelist):
        if (side in image):
            if (count == 0):
                pdf.add_page()
                count = 9
            count -= 1
            pdf.image(image, x=(x*cardW if (not reversed)
                      else cardW*(2-x)), y=y*cardH, w=cardW, h=cardH)
            x += 1
            if (x == 3):
                x = 0
                y += 1
                if (y == 3):
                    y = 0

    pdf.output(side+cType.capitalize()+"Template.pdf", "F")


def process(character, data, defaultStyle, frontColorTemplate, backColorTemplate, frontBlackTemplate, backBlackTemplate, offset):
    print(character["name"]+"...")
    image = character["image"].split("?")[0]
    content_path = load(image)
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

    stylized_image = stylized_image.convert("RGBA")

    width = imageSize
    height = stylized_image.size[1] * width//stylized_image.size[0]
    stylized_image = stylized_image.resize((width, height))

    words = character["name"].split(" ")
    name = "".join(words[:min(len(words), 2)])
    writeImages(stylized_image, "./art/color/"+name, character, offset,
                frontColorTemplate, backColorTemplate)
    writeImages(stylized_image, "./art/black/"+name, character, offset,
                frontBlackTemplate, backBlackTemplate)

    return name


def writeImages(image, fileName, character, offset, frontTemplate, backTemplate):
    color = "white" if ("color" in fileName) else "black"
    fnt = ImageFont.truetype(
        data["font"], 80-(floor(len(character["name"])/10))*8)
    fntSmall = ImageFont.truetype(data["font"], 60)
    traits = "\n".join(character["traits"])

    borderColor = borderColorDefault
    if ("borderColor" in character):
        borderColor = character["borderColor"]
    borderImage = border[borderColor]
    borderBackImage = borderBack[borderColor]

    front = frontTemplate.copy()
    Image.blend
    height = offset[1]+20
    height += drawText(front, character["name"],
                       (front.size[0]//2, height), fnt, color)
    if ("frontOffset" in character):
        height += character["frontOffset"]
    height += drawImage(image, (offset[0], height), front, color == "white")
    front.paste(borderImage, (0, 0), borderImage)
    front.save(fileName+"_front.png")

    back = backTemplate.copy()
    height = offset[1]+20
    height += drawText(back, character["name"],
                       (front.size[0]//2, height), fntSmall, color)
    if ("backOffset" in character):
        height += character["backOffset"]
    newSize=0.7
    smallImage = image.resize(
        (int(image.size[0]*newSize), int(image.size[1]*newSize)))
    height += drawImage(smallImage,
                        (offset[0]+(image.size[0]-smallImage.size[0])/2, height), back, color == "white")
    back.paste(borderBackImage, (0, 0), borderBackImage)
    drawText(back, traits,
             (back.size[0]//2, back.size[1]//1.47), fntSmall, "white")
    back.save(fileName+"_back.png")


if __name__ == "__main__":
    main()
