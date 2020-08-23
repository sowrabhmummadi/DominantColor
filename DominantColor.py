import numpy as np
import cv2

from ColorNode import ColorNode
from collections import deque

next_id = 1


def _get_leaves(node):
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


def _get_dominant_colors(root):
    leaves = _get_leaves(root)
    ret = []
    for leaf in leaves:
        mean = leaf.mean
        ret.append((mean[0][0] * 255.0, mean[1][0] * 255.0, mean[2][0] * 255.0))
    return ret


def _get_max_eigenvalue_node(current_node: ColorNode) -> ColorNode:
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


def _get_next_class_id(root):
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


class DominantColor:

    def __init__(self, image, no_of_palette_colors):
        self.image = image
        self.no_of_palette_colors = no_of_palette_colors
        self.rows, self.columns, _ = image.shape
        self.classes = np.ones((self.rows, self.columns, 1), dtype="uint8")

    def get_dominant_colors(self):
        root = ColorNode()
        root.class_id = 1
        self._get_class_mean_cov(root)
        for i in range(0, self.no_of_palette_colors - 1):
            next_node = _get_max_eigenvalue_node(root)
            self._partition_class(_get_next_class_id(root), next_node)
            self._get_class_mean_cov(next_node.left)
            self._get_class_mean_cov(next_node.right)

        return _get_dominant_colors(root)

    def _get_class_mean_cov(self, node):
        class_id = node.class_id
        sum_of_brg_values = np.zeros((3, 1))
        co_variance = np.zeros((3, 3))
        pixel_count = 1

        for y in range(0, self.rows):
            for x in range(0, self.columns):
                if self.classes[y, x] != class_id:
                    continue
                else:
                    brg_colors = self._get_bgr_colors(x, y)
                    sum_of_brg_values = sum_of_brg_values + brg_colors
                    co_variance = co_variance + (brg_colors * brg_colors.transpose())
                    pixel_count += 1
        cov = co_variance - (sum_of_brg_values * sum_of_brg_values.transpose()) / pixel_count
        mean = sum_of_brg_values / pixel_count
        node.mean = mean
        node.co_variance = cov

    def _get_bgr_colors(self, x, y):
        color = self.image[y, x]
        bgr_colors = np.zeros((3, 1))
        # converting the RGB values within the  0...1 to reduce the chance of overflow
        bgr_colors[0] = color[2] / 255
        bgr_colors[1] = color[1] / 255
        bgr_colors[2] = color[0] / 255
        return bgr_colors

    def _partition_class(self, next_id, node: ColorNode):
        mean = node.mean
        class_id = node.class_id
        ret_val, eigenvalues, eigenvectors = cv2.eigen(node.co_variance)
        comparison_value = eigenvectors[0] * mean

        node.left = ColorNode()
        node.left.class_id = next_id
        node.right = ColorNode()
        node.right.class_id = next_id + 1

        for y in range(0, self.rows):
            cp = self.classes[y]
            for x in range(0, self.columns):
                if self.classes[y, x] != class_id:
                    continue
                else:
                    brg_colors = self._get_bgr_colors(x, y)
                    current_value = eigenvectors[0] * brg_colors

                    if current_value[0, 0] <= comparison_value[0, 0]:
                        cp[x] = next_id
                    else:
                        cp[x] = next_id + 1
