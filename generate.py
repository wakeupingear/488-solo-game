import tensorflow_hub as hub
import tensorflow as tf
from simplejson import load
import urllib.request
from urllib.parse import urlparse
from ast import arg
from PIL import ImageColor, Image, ImageDraw, ImageFont
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

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


def drawText(image, msg, offset, fnt, fill="white", maxWidth=700):
    draw = ImageDraw.Draw(image)
    msg = wrap_text(msg, maxWidth, fnt)
    height = 0
    for i, line in enumerate(msg):
        w, h = draw.textsize(line, font=fnt)
        height = (h*1.4)*(i)
        draw.text((-w//2+offset[0], h//2+offset[1] +
                  height), line, fill=fill, font=fnt)
    return height


def main():
    print("Loading characters...")
    f = open("characters.json", "r+")
    data = json.load(f)

    defaultStyle = load(data["styles"][0])
    widthHalved = (data["width"] - imageSize)//2
    offset = (widthHalved, widthHalved-64)

    bgColor = ImageColor.getcolor(data["bgColor"], "RGBA")
    borderColor = ImageColor.getcolor(data["borderColor"], "RGBA")
    frontTemplate = Image.new("RGBA", (data["width"], data["height"]), bgColor)
    backTemplate = frontTemplate.copy()
    draw = ImageDraw.Draw(frontTemplate)
    draw.rectangle((offset[0]-32, offset[1]-32, offset[0] +
                   imageSize+32, offset[1]+imageSize+32), borderColor)

    if not os.path.exists("art"):
        print("Creating new art directory")
        os.makedirs("art")

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
                            frontTemplate, backTemplate, offset)
                    found = True
                    break
            if not found:
                print("Character not found")
    else:
        for character in data["characters"]:
            process(character, data, defaultStyle,
                    frontTemplate, backTemplate, offset)


saveDir = "art/"


def process(character, data, defaultStyle, frontTemplate, backTemplate, offset):
    print(character["name"]+"...")
    image = character["image"]
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

    front = frontTemplate.copy()
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 80)
    drawText(front, character["name"],
             (front.size[0]//2, imageSize+100), fnt)
    front.paste(stylized_image, (offset[0], offset[1]))

    back = backTemplate.copy()
    height = drawText(back, character["name"], (back.size[0]//2, 40), fnt)
    traits = "\n".join(character["traits"])
    drawText(back, traits, (back.size[0]//2, height+300), fnt)

    words = character["name"].split(" ")
    filename = "".join(words[:min(len(words), 2)])
    front.save(saveDir+filename+"_front.png")
    back.save(saveDir+filename+"_back.png")


if __name__ == "__main__":
    main()
