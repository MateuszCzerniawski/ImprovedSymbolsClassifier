import os
import cv2
import numpy as np
import pandas as pd

symbols_names = ['ankha', 'loop', 'p', 'grass',
                 'rect', 'vawe', 'plate', 'eye',
                 'circ_center', 'circ_stripped', 'circ_x', 'scarab']


def detect_grid_edges(img):
    height, width, _ = img.shape
    h_pos, v_pos = [0], [0]
    edge = False
    for x in range(width):
        pixel = img[0][x]
        if not edge and pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            last = len(h_pos) - 1
            h_pos[last] = (h_pos[last], x - 1)
            edge = True
        if edge and pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            h_pos.append(x)
            edge = False
    last = len(h_pos) - 1
    h_pos[last] = (h_pos[last], width)
    for y in range(height):
        pixel = img[y][0]
        if not edge and pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            last = len(v_pos) - 1
            v_pos[last] = (v_pos[last], y - 1)
            edge = True
        if edge and pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            v_pos.append(y)
            edge = False
    last = len(v_pos) - 1
    v_pos[last] = (v_pos[last], height)
    return h_pos, v_pos


def slice_image(img):
    images = []
    h_edges, v_edges = detect_grid_edges(img)
    for h in h_edges[1:]:  # border skip
        for v in v_edges[2:]:  # border and perfect skip
            images.append(img[v[0]:v[1], h[0]:h[1]])
    return images


def invert(image):
    image = image.copy()
    if (len(image.shape)) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    for x in range(width):
        for y in range(height):
            image[y][x] = 255 - image[y][x]
    return image


def shift_image(image, shift):
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    dsize = (image.shape[1], image.shape[0])
    return cv2.warpAffine(image, matrix, dsize, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def shift_to_center(image):
    mask = invert(image)
    M = cv2.moments(mask)
    x, y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    height, width = mask.shape
    return shift_image(image, (width // 2 - x, height // 2 - y))


def minimise(image, scale):
    scale = 1 / scale if scale < 1 else scale
    height, width, _ = image.shape
    width, height = int(width / scale), int(height / scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def image_to_row(image):
    row = []
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for i in image.flatten():
            row.append(1 if i == 255 else 0)
    return row


def images_to_dataframe(images):
    return pd.DataFrame([image_to_row(i) for i in images])


def process_raw_data(input_dir, output_dir, scales=()):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    all_images = []
    labels = []
    for path in [f'{input_dir}/{file}' for file in os.listdir(input_dir)]:
        img = cv2.imread(path)
        sliced = slice_image(img)
        sliced = [shift_to_center(i) for i in sliced]
        all_images.extend(sliced)
        labels.extend([i // len(symbols_names) for i in range(len(sliced))])
    images_to_dataframe(all_images).to_csv(f'{output_dir}/x', index=False)
    pd.DataFrame(labels).to_csv(f'{output_dir}/y', index=False)
    if len(scales) > 0:
        for scale in scales:
            minimised = [minimise(i, scale) for i in all_images]
            images_to_dataframe(minimised).to_csv(f'{output_dir}/x{scale}', index=False)


def load(data_path, labels_path):
    x = pd.DataFrame(pd.read_csv(data_path))
    x.reset_index(drop=True, inplace=True)
    y = pd.DataFrame(pd.read_csv(labels_path))
    y.reset_index(drop=True, inplace=True)
    return x, y


# process_raw_data('../data/images/raw', '../data/in_csv', scales=range(2, 21))
# x, y = load('../data/in_csv/x', '../data/in_csv/y')
# print(x.shape)
# for index, row in x.iterrows():
#     print(f'{index}: {row.shape}')
#process_raw_data('../data/extended/', '../data/extended_csv', scales=range(2, 21))