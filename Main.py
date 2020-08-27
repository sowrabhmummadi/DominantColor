import argparse

import cv2
import numpy
import json

from os import path,remove
from DominantColor import DominantColor
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def main():

    parser = argparse.ArgumentParser(description='Find dominant colors in a image')
    parser.add_argument('-p', '--image_path', help="path of image to get dominant colors form",
                        type=str)
    parser.add_argument('-npc', '--number_of_palette_colors', help="number of dominant colors to identify",
                        type=int)
    parser.add_argument('-o', '--output_file_name', help="output filename for dominant colors palette png file",
                        type=str)
    args = parser.parse_args()

    if not path.exists(args.image_path):
        raise Exception("image path is not available")
    else:
        filename, extension = path.splitext(args.image_path)
        if extension == ".svg":
            print("given an svg file converting it to png")
            drawing = svg2rlg(args.image_path)
            renderPM.drawToFile(drawing, f"{filename}.png", fmt="PNG")
            args.image_path = f"{filename}.png"
            print(f"Done-created new file: {filename}.png")

    if 0 <= args.number_of_palette_colors and args.number_of_palette_colors > 255:
        raise Exception("Number should be in the range: 0....<number>....255")
    cv_image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    remove(args.image_path)
    colors = DominantColor(cv_image, args.number_of_palette_colors).get_dominant_colors()
    palette_colors = dict()

    for i in range(0, len(colors)):
        (r, g, b) = colors[i]
        color_info = dict()
        color_info['rgb'] = (r, g, b)
        color_info['hex'] = '#{:02x}{:02x}{:02x}'.format(int(numpy.rint(r).item()),
                                                         int(numpy.rint(g).item()),
                                                         int(numpy.rint(b).item()))
        palette_colors[f"color:{i}"] = color_info

    cv2.imwrite(f"{args.output_file_name}.png", get_dominant_palette(colors))
    with open(f"{args.output_file_name}.json", "w+") as file:
        file.write(json.dumps(palette_colors))
        file.close()




def get_dominant_palette(colors):
    palette_tile_size = 200
    rect = numpy.zeros([palette_tile_size, palette_tile_size * len(colors), 3], numpy.uint8)

    for index, color in enumerate(colors):
        r, g, b = color
        cv2.rectangle(rect, (index * palette_tile_size, 0),
                      ((index + 1) * palette_tile_size, (index + 1) * palette_tile_size), (b, g, r),
                      thickness=-1)
    return rect


if __name__ == "__main__":
    main()
