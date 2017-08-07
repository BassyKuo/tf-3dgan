#!/usr/bin/env python3
# REFER TO https://github.com/nmhkahn/DCGAN-tensorflow-slim.git
import os
import glob
import wget
import tarfile
import zipfile
import argparse
import scipy.misc
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-dataset", "-D",
                        type=str,
                        choices=["celebA", "flowers"],
                        default="celebA",
                        help="Name of dataset to download and convert \
                                [default: %(default)s]")
    parser.add_argument("--format", "-format", "-F",
                        type=str,
                        choices=["npy", "tfrecords"],
                        default="npy",
                        help="Dataset format to save [default: %(default)s]")
    return parser.parse_args()


def download_and_convert_flowers(filename):
    LABELS = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    url = "http://download.tensorflow.org/example_images/flower_photos.tgz"

    if not os.path.exists("flower_photos"):
        print("Download flower dataset..")
        wget.download(url)
        print("\nExtracting dataset..")
        tarfile.open("flower_photos.tgz", "r:gz").extractall("./")
        os.remove("flower_photos.tgz")

    if filename.split('.')[-1] == 'tfrecords':
        print("Convert dataset to TFRecord format..")
        writer = tf.python_io.TFRecordWriter(filename)
        for label in LABELS:
            paths = glob.glob(os.path.join("flower_photos", label)+"/*.jpg")
            for path in paths:
                im = scipy.misc.imread(path)
                im = scipy.misc.imresize(im, [64, 64])

                h, w = im.shape[:2]
                im_raw = im.tostring()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                            "height": _int64_feature([h]),
                            "width": _int64_feature([w]),
                            "image": _bytes_feature([im_raw])
                        }))
                writer.write(example.SerializeToString())
        writer.close()
    else:
        print("Convert dataset to npy format..")
        images = []
        for label in LABELS:
            paths = glob.glob(os.path.join("flower_photos", label)+"/*.jpg")
            for path in paths:
                im = scipy.misc.imread(path)
                im = scipy.misc.imresize(im, [64, 64])
                images.append(im)
        images = np.asarray(images)
        np.save(filename, images)


def download_and_convert_celeba(filename):
    url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/"+ \
          "AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1"

    if not os.path.exists("img_align_celeba"):
        print("Download celebA dataset..")
        wget.download(url)
        print("\nExtracting dataset..")
        zipfile.ZipFile("img_align_celeba.zip").extractall("./")
        os.remove("img_align_celeba.zip")

    def center_crop( im,
                     output_height,
                     output_width ):
        h, w = im.shape[:2]
        if h < output_height and w < output_width:
            raise ValueError("image is small")

        offset_h = int((h - output_height) / 2)
        offset_w = int((w - output_width) / 2)
        return im[offset_h:offset_h+output_height,
                  offset_w:offset_w+output_width, :]

    paths = glob.glob("img_align_celeba/*.jpg")

    if filename.split('.')[-1] == 'tfrecords':
        print("Convert dataset to TFRecord format..")
        writer = tf.python_io.TFRecordWriter(filename)
        for i, path in enumerate(paths):
            im = scipy.misc.imread(path)
            im = center_crop(im, 128, 128)
            im = scipy.misc.imresize(im, [64, 64])

            h, w = im.shape[:2]
            im_raw = im.tostring()

            example = tf.train.Example(features=tf.train.Features(
                feature={
                        "height": _int64_feature([h]),
                        "width": _int64_feature([w]),
                        "image": _bytes_feature([im_raw])
                    }))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print("Convert dataset to npy format..")
        images = np.zeros([len(paths), 64, 64, 3])
        for i, path in enumerate(paths):
            im = scipy.misc.imread(path)
            im = center_crop(im, 128, 128)
            im = scipy.misc.imresize(im, [64, 64])
            images[i] = im
        np.save(filename, images)
        

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "celebA":
        if args.format == "tfrecords":
            download_and_convert_celeba("celeba.tfrecords")
        else:
            download_and_convert_celeba("celeba_64")
    elif args.dataset == "flowers":
        if args.format == "tfrecords":
            download_and_convert_flowers("flowers.tfrecords")
        else:
            download_and_convert_flowers("flowers_64")
