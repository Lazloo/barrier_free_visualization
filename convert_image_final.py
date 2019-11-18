import colorsys
from tqdm import tqdm
import ast
import operator
from functools import reduce
from PIL import Image
import matplotlib.pyplot as plt

plt.rcdefaults()

def QuantizeToGivenPalette(im, palette):
    """Quantize image to a given palette.

    The input image is expected to be a PIL Image.
    The palette is expected to be a list of no more than 256 R,G,B values."""

    e = len(palette)
    assert e > 0, "Palette unexpectedly short"
    assert e <= 768, "Palette unexpectedly long"
    assert e % 3 == 0, "Palette not multiple of 3, so not RGB"

    # Make tiny, 1x1 new palette image
    p = Image.new("P", (1, 1))

    # Zero-pad the palette to 256 RGB colours, i.e. 768 values and apply to image
    palette += (768 - e) * [0]
    p.putpalette(palette)

    # Now quantize input image to the same palette as our little image
    return im.convert("RGB").quantize(palette=p)



file_name = 'bunte_bretter'
file_name = '1'
im = Image.open('data/' + file_name + '.jpg').convert('RGB')
indices = [331, 667, 908, 946, 1134, 1301, 2301]
indices = [0]
RGB_tuples_proto = [
    (17, 112, 170),
    (200, 82, 0),
    (252, 125, 11),
    (123, 132, 143),
    (163, 172, 185),
    (163, 204, 233),
    (87, 96, 108),
    (255, 188, 121),
    (95, 162, 206),
    (200, 208, 217),
    (0, 0, 0),
    (255, 255, 255)
]
RGB_tuples_proto = [tuple([j/255.0 for j in i]) for i in RGB_tuples_proto]
RGB_tuples = RGB_tuples_proto
# with open("file.txt", "r") as output:
#     RGB_tuples_proto = output.read()

for i in indices:
    RGB_tuples = ast.literal_eval(RGB_tuples_proto)
    # RGB_tuples = RGB_tuples[918]
    RGB_tuples = RGB_tuples[i]
        # RGB_tuples.append((0, 0, 0))
        # RGB_tuples.append((1, 1, 1))

    newPalette = [int(i * 255) for i in list(reduce(operator.add, RGB_tuples))]

    r = QuantizeToGivenPalette(im, newPalette)
    # Save result
    r.save('result/' + file_name + str(i)+'.png')
