import argparse
from collections import deque
from os import path

import cv2
import numpy

from ColorNode import ColorNode


def main():
    parser = argparse.ArgumentParser(description='Find dominant colors in a image')
    parser.add_argument('-p', '--image_path', help="path of image to get dominant colors form",
                        type=str)
    parser.add_argument('-ndc', '--required_number_of_colors', help="number of dominant colors to identify",
                        type=int)
    parser.add_argument('-o', '--output', help="output filename for dominant colors palette png file", type=str)
    args = parser.parse_args()

    if not path.exists(args.image_path):
        raise Exception("image path is not available")
    if 0 <= args.required_number_of_colors and args.required_number_of_colors > 255:
        raise Exception("Number should be in the range: 0....<number>....255")
    cv_image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    colors = find_dominant_colors(cv_image, args.required_number_of_colors)
    print(colors)
    cv2.imwrite(f"{args.output}.png", get_dominant_palette(colors))


def get_dominant_palette(colors):
    palette_tile_size = 200
    rect = numpy.zeros([palette_tile_size, palette_tile_size * len(colors), 3], numpy.uint8)

    for index, color in enumerate(colors):
        r,g,b = color
        cv2.rectangle(rect, (index * palette_tile_size, 0),
                      ((index+1)*palette_tile_size, (index+1)*palette_tile_size), (b, g, r),
                      thickness=-1)
    return rect


def get_next_class_id(root):
    max_id = 0
    queue = deque()
    queue.append(root)

    while len(queue) > 0:
        current = queue.popleft()
        if current.class_id > max_id:
            max_id = current.class_id
        if current.left:
            queue.append(current.left)
        if current.right:
            queue.append(current.right)

    return max_id + 1


def partition_class(image, classes, next_id, node: ColorNode):
    height, width, _ = image.shape
    class_id = node.class_id
    mean = node.mean
    ret_val, eigenvalues, eigenvectors = cv2.eigen(node.co_variance)
    comparison_value = eigenvectors[0] * mean

    node.left = ColorNode()
    node.left.class_id = next_id
    node.right = ColorNode()
    node.right.class_id = next_id + 1

    for y in range(0, height):
        cp = classes[y]
        for x in range(0, width):
            if classes[y, x] != class_id:
                continue
            else:
                scaled_rgb = get_scaled_rgb(image, x, y)

                current_value = eigenvectors[0] * scaled_rgb

                if current_value[0, 0] <= comparison_value[0, 0]:
                    cp[x] = next_id
                else:
                    cp[x] = next_id + 1


def get_scaled_rgb(image, x, y):
    color = image[y, x]
    scaled = numpy.zeros((3, 1))
    # converting the RGB values within the  0...1 to reduce the chance of overflow
    scaled[0] = color[2] / 255
    scaled[1] = color[1] / 255
    scaled[2] = color[0] / 255
    return scaled


def get_max_eigenvalue_node(current_node: ColorNode) -> ColorNode:
    max_eigen = -1
    queue = deque()
    queue.append(current_node)

    if not current_node.left and not current_node.right:
        return current_node
    ret = None
    while len(queue) > 0:
        node = queue.popleft()
        if node.left and node.right:
            queue.append(node.left)
            queue.append(node.right)
            continue
        ret_val, eigenvalues, eigenvectors = cv2.eigen(current_node.co_variance)
        eigenvalue = eigenvalues[0]
        if max_eigen < eigenvalue:
            max_eigen = eigenvalue
            ret = node
    return ret


def get_leaves(node):
    queue = deque()
    leaves = []
    queue.append(node)

    while len(queue) > 0:
        node = queue.popleft()
        if node.left and node.right:
            queue.append(node.left)
            queue.append(node.right)
            continue
        leaves.append(node)

    return leaves


def get_dominant_colors(root):
    leaves = get_leaves(root)
    ret = []
    for leaf in leaves:
        mean = leaf.mean
        ret.append((mean[0][0] * 255.0, mean[1][0] * 255.0, mean[2][0] * 255.0))
    return ret


def find_dominant_colors(image, no_of_colors):
    rows, columns, _ = image.shape
    classes = numpy.ones((rows, columns, 1), dtype="uint8")

    root = ColorNode()
    root.class_id = 1
    get_class_mean_cov(image, classes, root)
    for i in range(0, no_of_colors - 1):
        next_node = get_max_eigenvalue_node(root)
        partition_class(image, classes, get_next_class_id(root), next_node)
        get_class_mean_cov(image, classes, next_node.left)
        get_class_mean_cov(image, classes, next_node.right)

    return get_dominant_colors(root)


def get_class_mean_cov(image, classes, node):
    height, width, _ = image.shape
    class_id = node.class_id

    mean = numpy.zeros((3, 1))

    co_variance = numpy.zeros((3, 3))
    pixel_count = 1

    for y in range(0, height):
        for x in range(0, width):
            if classes[y, x] != class_id:
                continue
            else:
                scaled_rgb = get_scaled_rgb(image, x, y)
                mean = mean + scaled_rgb
                co_variance = co_variance + (scaled_rgb * scaled_rgb.transpose())
                pixel_count += 1
    cov = co_variance - (mean * mean.transpose()) / pixel_count
    mean = mean / pixel_count
    node.mean = mean
    node.co_variance = cov


if __name__ == "__main__":
    main()
