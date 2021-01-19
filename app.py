#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 miljanic <miljanic.m@protonmail.com>
#
# Distributed under terms of the BSD 3-Clause license.
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import cv2
import argparse


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


hsv_colors = {
    "red": [
        (np.array([0, 70, 70]),
         np.array([3, 255, 210])),
        (np.array([175, 70, 70]),
         np.array([180, 255, 210]))
    ],
    "yellow": [
        (np.array([18, 70, 50]),
         np.array([48, 255, 255]))
    ],
    "blue": [
        (np.array([92, 70, 30]),
         np.array([131, 255, 200]))
    ],
    "green": [
        (np.array([60, 70, 24]),
         np.array([91, 255, 130]))
    ],
    "white": [
        (np.array([0, 0, 100]),
         np.array([255, 53, 255]))
    ],
    "orange": [
        (np.array([4, 70, 50]),
         np.array([17, 255, 255]))
    ],
}


@dataclass
class Square:
    left: float
    bottom: float
    right: float
    top: float

    @classmethod
    def normalize(cls, x1, y1, x2, y2) -> 'Square':
        return Square(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    @property
    def width(self) -> float:
        return max(self.right - self.left, 0)

    @property
    def height(self) -> float:
        return max(self.top - self.bottom, 0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def __bool__(self):
        return bool(self.area > 0)

    def intersect(self, other: 'Square') -> 'Square':
        return Square(
            max(self.left, other.left),
            max(self.bottom, other.bottom),
            min(self.right, other.right),
            min(self.top, other.top)
        )


def get_squares_from_frame(img):
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 \
                        and cv2.contourArea(cnt) > 1000 \
                        and cv2.contourArea(cnt) < 10000 \
                        and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def is_last_from_intersection(big: Square, squares: List[Square]) -> bool:
    """Checks that square doesn't have any intersections with other squares"""
    for s in squares:
        if s.intersect(big):
            return False
    return True


class CubeState:
    def __init__(self, cube_dimensions: int):
        self.squares_count = cube_dimensions ** 2

    def all_squares_found(self, squares) -> bool:
        return len(squares) == self.squares_count


def filter_intersecting_squares(squares: List[Square]) -> List[Square]:
    for_drawing = []
    for index, s in enumerate(squares):
        if is_last_from_intersection(s, squares[index + 1:]):
            for_drawing.append(s)
    return for_drawing


def main(video: Union[str, int], cube_dimensions: int):
    cap = cv2.VideoCapture(video)
    state = CubeState(cube_dimensions)

    while True:
        # get single frame from video
        ret, image = cap.read()
        img = cv2.GaussianBlur(image, (5, 5), 0)

        # extract squares from image
        squares = [
            Square.normalize(s[0][0], s[0][1], s[2][0], s[2][1])
            for s in get_squares_from_frame(image)
        ]
        for_drawing = filter_intersecting_squares(squares)

        # draw
        thickness = 4 if state.all_squares_found(for_drawing) else 1
        color = (0, 266, 0) if state.all_squares_found(for_drawing) else (0, 0, 255)

        def get_square_color(full_image, square):
            avg_b = 0
            avg_g = 0
            avg_r = 0
            pixel_count = (square.right - square.left) * (square.top - square.bottom)
            for x in range(int(c.left), int(c.right), 1):
                for y in range(int(c.bottom), int(c.top), 1):
                    pixel_color = full_image[y, x]
                    avg_b += pixel_color[0]
                    avg_g += pixel_color[1]
                    avg_r += pixel_color[2]
            avg_color = np.uint8([[[avg_b / pixel_count, avg_g / pixel_count, avg_r / pixel_count]]])
            # bgr to hsv
            avg_hsv = cv2.cvtColor(avg_color, cv2.COLOR_BGR2HSV)

            search = avg_hsv[0][0]
            for col_name, vals in hsv_colors.items():
                for low, high in vals:
                    if low[0] <= search[0] <= high[0] and low[1] <= search[1] <= high[1] and low[2] <= search[2] <= \
                            high[2]:
                        return col_name
            return None

        found_colors = []
        for c in for_drawing:
            cv2.rectangle(
                image,
                pt1=(c.left, c.bottom),
                pt2=(c.right, c.top),
                color=color,
                thickness=thickness
            )
            if len(for_drawing) == state.squares_count:
                found_colors.append(get_square_color(img, c))
                found_colors = list(filter(lambda x: x, found_colors))

        if len(for_drawing) == state.squares_count == len(found_colors):
            print('TODO: steps')

        cv2.imshow("Image", image)
        k = cv2.waitKey(1) & 0xFF
        # press escape to exit
        if k == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='Path to a video file.')
    parser.add_argument('--sides', help='Number of cube sides.')
    args = parser.parse_args()

    video = args.video if args.video else 0
    cube_dimensions = int(args.sides) if args.sides else 3

    main(video, cube_dimensions)
