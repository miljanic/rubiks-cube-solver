#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 miljanic <miljanic.m@protonmail.com>
#
# Distributed under terms of the BSD 3-Clause license.
from dataclasses import dataclass
from typing import NamedTuple, List, Union

import numpy as np
import cv2
import argparse


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


class Point(NamedTuple):
    x: float
    y: float


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


def main(video: Union[str, int], cube_dimensions: int):
    cap = cv2.VideoCapture(video)
    state = CubeState(cube_dimensions)

    while True:
        ret, image = cap.read()

        img = cv2.GaussianBlur(image, (5, 5), 0)

        squares = [
            Square.normalize(s[0][0], s[0][1], s[2][0], s[2][1])
            for s in get_squares_from_frame(img)
        ]

        for_drawing = []

        for index, s in enumerate(squares):
            if is_last_from_intersection(s, squares[index + 1:]):
                for_drawing.append(s)

        thickness = 4 if state.all_squares_found(for_drawing) else 1
        color = (0, 266, 0) if state.all_squares_found(for_drawing) else (0, 0, 255)

        for c in for_drawing:
            cv2.rectangle(
                image,
                pt1=(c.left, c.bottom),
                pt2=(c.right, c.top),
                color=color,
                thickness=thickness
            )

        cv2.imshow("Image", image)
        cv2.waitKey()

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
