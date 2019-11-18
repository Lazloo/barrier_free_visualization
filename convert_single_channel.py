from PIL import Image
from tqdm import tqdm
import numpy as np

# Load and prepare image array
file_name = '1'
# file_name = 'bunte_bretter'
file_name = 'TCFruits_original'
im = Image.open('data/' + file_name + '.jpg').convert('RGB')
pixels = np.array(im)
s = pixels.shape
p = pixels.reshape((s[0]*s[1], s[2]))

# for i, rgb in tqdm(enumerate(p)):
#     # if rgb[1] > rgb[2]:
#         # p[i] = [rgb[0], rgb[1], rgb[1]]
#     p[i] = [rgb[0], max(rgb[1] - rgb[0], 0), rgb[1]]

# p = np.array([[rgb[0], max(rgb[1] - rgb[0]*0.0, 0), rgb[1]] for i, rgb in tqdm(enumerate(p))])
p = np.array([[rgb[0], max(rgb[1] - rgb[1]*0.5, 0), rgb[2]] for i, rgb in tqdm(enumerate(p))])
    # [[p[0], p[1], p[2]] ]
# Map to discrete color space
# RGB_tuples_converted = np.array(list(map(lambda x: tuple(map_rgb(x)), p)))
#
# # Generate Image
img = Image.fromarray(np.uint8(p.reshape(s)))
img.save('result/' + file_name + '_test.png')