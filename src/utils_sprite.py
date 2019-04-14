import os
import pickle
from typing import List

import numpy as np
import cv2
import glob


BIN_FILE = 'bin_sprites'
BIN_LOW_RES_FILE = "bin_low_res_sprites"


def load_sprites_from_png() -> List[np.ndarray]:
    img_l = []
    for file in glob.iglob('../sprites/*.png'):
        img = cv2.imread(file)
        img_l.append(img[..., :3])

    return img_l


def save_sprites(file: str, img_l: List[np.ndarray]):
    if os.path.isfile(file):
        ans = input("File already exists, want to erase ? (y/n)")
        if ans != 'y' and ans != "Y":
            return

    print(f"Writing file {file}")
    with open(file, 'wb') as fp:
        pickle.dump(img_l, fp)


def load_sprites_from_bin(file: str) -> List[np.ndarray]:
    if not os.path.isfile(file):
        print(f"File {file} does not exist")
        return []
    try:
        with open(file, 'rb') as fp:
            return pickle.load(fp)
    except pickle.UnpicklingError:
        print(f"Error while unpickling {file}")
    except IOError:
        print(f"Error while unpickling {file}")


def main():

    # img_l = load_sprites_from_png()
    # save_sprites(BIN_FILE, img_l)

    img_l = load_sprites_from_bin(BIN_FILE)

    print(img_l[0].shape)

    img_l = list(map(lambda i: cv2.resize(i, (32, 32)), img_l))

    cv2.imshow('bulbasaur', img_l[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_sprites(BIN_LOW_RES_FILE, img_l)

    pass


if __name__ == '__main__':
    main()
