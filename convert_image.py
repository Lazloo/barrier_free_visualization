import colorsys
from tqdm import tqdm
import operator
from functools import reduce
from PIL import Image
from scipy.spatial import distance
import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


def get_palette():
    N = np.random.randint(2, 253 + 1)
    # HSV_tuples = [(x * 1.0 / (N - 1), 0.5, 0.8) for x in range(N)]  # 0 ... red, 0.333 .... green, 0.6666 ... blue
    # HSV_tuples = [(x * 1.0 / (N - 1) / 3 * 2, 0.75, 0.6) for x in range(N)]  # 0 ... red, 0.333 .... green, 0.6666 ... blue
    HSV_tuples = [tuple(np.random.uniform(size=3)) for i in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    RGB_tuples.append((0, 0, 0))
    RGB_tuples.append((1, 1, 1))
    return RGB_tuples


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


im = Image.open('data/1.jpg').convert('RGB')

rgb_list = []
for i in tqdm(range(0, 10000)):
    RGB_tuples = get_palette()
    inPalette = [int(i * 255) for i in list(reduce(operator.add, RGB_tuples))]
    newPalette = [int(i * 255) for i in list(reduce(operator.add, RGB_tuples))]
    rgb_list.append(RGB_tuples)

    r = QuantizeToGivenPalette(im, newPalette)
    # Save result
    r.save('test/result' + str(i) + '.png')

with open("file.txt", "w") as output:
    output.write(str(rgb_list))
#
# p = [i / 255.0 for i in r.getpalette()]
# p = np.array(p).reshape(int(len(p) / 3), 3)
# #
# objects = [tuple(i) for i in p]
# n = len(RGB_tuples)
# y_pos = np.arange(n - 2)
# performance = [1] * len(p)
#
# #
# plt.figure(figsize=(16, 9))
# plt.bar(y_pos[0:n - 2], performance[0:n - 2], align='center', alpha=1, color=objects[0:n - 2])
# plt.xticks(y_pos, y_pos)
# plt.show()
