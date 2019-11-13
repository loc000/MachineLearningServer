from __future__ import print_function
from PIL import Image
import os
import argparse


def resize_image(image, image_size):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([image_size, image_size], Image.ANTIALIAS)
    return image

def main(use_inception):
    splits = ['train', 'val']
    if use_inception:
        image_size = 299
    else:
        image_size = 224
    for split in splits:
        folder = './image/%s2014' %split
        resized_folder = './image/%s2014_resized/' %split
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        print('Start resizing {} images {} * {}.'.format(split, image_size, image_size))
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image, image_size)
                    image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print('Resized images: {}/{}'.format(i, num_images))
              
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_inception", action="store_true", help="use inception network image size (299 * 299)")
    args = parser.parse_args()
    main(args.use_inception)
