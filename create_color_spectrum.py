import colorsys
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


N = 30
HSV_tuples = [(x*1.0/(N-1)/3*2, 1, 0.75) for x in range(N)] # 0 ... red, 0.333 .... green, 0.6666 ... blue
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))


print(colorsys.rgb_to_hsv(0,0,1))

objects = tuple([i for i in range(N)])
y_pos = np.arange(len(objects))
performance = [1]*N


plt.figure(figsize=(16, 9))
plt.bar(y_pos, performance, align='center', alpha=0.5, color=RGB_tuples)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')
plt.show()
# 1, 3, 6, 10, 14,17, 20, 22, 24, 26, 29
# 0,2,4,6, 12, 17,19,21,23,25, 26, 29

indices_good_colors = [0, 2, 4, 7, 12, 17, 19, 21, 23, 25, 26, 29]
n_new_color = len(indices_good_colors)
RGB_tuples_selected = [RGB_tuples[i] for i in indices_good_colors]
objects_selected = tuple([i for i in range(n_new_color)])
y_pos_selected = np.arange(len(objects_selected))

plt.figure(figsize=(16, 9))
plt.bar(y_pos_selected, [1]*n_new_color, align='center', alpha=0.5, color=RGB_tuples_selected)
plt.xticks(y_pos_selected, objects_selected)
plt.ylabel('Usage')
plt.title('Programming language usage')
plt.show()

hsv_select_color = [colorsys.rgb_to_hsv(i_rgb[0], i_rgb[1], i_rgb[2])[0] for i_rgb in RGB_tuples_selected]

def map_rgb(rgb_value: list):
    hsv_value = colorsys.rgb_to_hsv(rgb_value[0], rgb_value[1], rgb_value[2])
    min_dist_index = np.argmin([abs(i_hsv - hsv_value[0]) for i_hsv in hsv_select_color])
    hsv_value_new = colorsys.hsv_to_rgb(hsv_select_color[min_dist_index], hsv_value[1], hsv_value[2])
    return np.array(hsv_value_new)


RGB_tuples_converted = list(map(lambda x: map_rgb(x), RGB_tuples))
plt.figure(figsize=(16, 9))
plt.bar(y_pos, performance, align='center', alpha=0.5, color=RGB_tuples_converted)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')
plt.show()


from PIL import Image
jpgfile = Image.open("data/red_green_test.jpg")
pixels = np.array(jpgfile)

pixels_new = pixels
for i in range(len(pixels)):
    for j in range(len(pixels[0])):
        pixels_new[i][j] = map_rgb(pixels_new[i][j])

img = Image.fromarray(pixels_new)
# img = img.convert('L')
img.save('data/myimg.jpeg')