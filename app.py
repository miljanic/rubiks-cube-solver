#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 miljanic <miljanic.m@protonmail.com>
#
# Distributed under terms of the BSD 3-Clause license.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import cv2  # type: ignore
from rubik_solver import utils  # type: ignore

IndexedColors = Dict[Tuple[int, int], Optional[str]]
KnownIndexedColors = Dict[Tuple[int, int], str]


class VideoDone(Exception):
    pass


class AllSidesFound(Exception):
    pass


class UnknownColorFound(Exception):
    pass


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


hsv_colors = {
    "red": [
        (np.array([0, 70, 70]),
         np.array([3, 255, 255])),
        (np.array([173, 70, 70]),
         np.array([180, 255, 255]))
    ],
    "yellow": [
        (np.array([18, 70, 50]),
         np.array([48, 255, 255]))
    ],
    "blue": [
        (np.array([92, 70, 30]),
         np.array([131, 255, 255]))
    ],
    "green": [
        (np.array([60, 70, 24]),
         np.array([91, 255, 255]))
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
    right: float
    top: float
    bottom: float

    @classmethod
    def normalize(cls, x1, y1, x2, y2) -> Square:
        return Square(
            left=min(x1, x2),
            right=max(x1, x2),
            top=min(y1, y2),
            bottom=max(y1, y2),
        )

    @property
    def width(self) -> float:
        return max(self.right - self.left, 0)

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 0)

    @property
    def area(self) -> float:
        return abs(self.width * self.height)

    def __bool__(self) -> bool:
        return bool(self.area > 0)

    def intersect(self, other: Square) -> Square:
        return Square(
            left=max(self.left, other.left),
            right=min(self.right, other.right),
            top=max(self.top, other.top),
            bottom=min(self.bottom, other.bottom),
        )


def get_squares_from_frame(img):
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 250, 40):
            if thrs == 0:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                black_white = cv2.Canny(gray, 60, 100)
                black_white = cv2.dilate(black_white, None)
            else:
                _, black_white = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                black_white,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.15 * cnt_len, True)
                if (
                    len(cnt) == 4
                    and cv2.contourArea(cnt) > 1000
                    and cv2.contourArea(cnt) < 10000
                    and cv2.isContourConvex(cnt)
                ):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([
                        angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                        for i in range(4)
                    ])
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
    def __init__(self):
        self.dimension = 3
        self.squares_count = self.dimension ** 2
        self.sides: Dict[str, KnownIndexedColors] = {}

    def all_squares_found(self, squares) -> bool:
        return len(squares) == self.squares_count

    def all_sides_found(self) -> bool:
        return len(self.sides) == 6

    def add_side(self, colors: KnownIndexedColors) -> None:
        middle_piece = colors[(1, 1)]
        if middle_piece not in self.sides:
            print(f'Adding {middle_piece} side')
            self.sides[middle_piece] = colors

    def get_pieces_of_side(self, side: str) -> Iterator[str]:
        for piece in self.sides[side].values():
            yield piece[0].lower()

    def get_pieces_in_order(self) -> Iterator[str]:
        yield from self.get_pieces_of_side('yellow')
        yield from self.get_pieces_of_side('blue')
        yield from self.get_pieces_of_side('red')
        yield from self.get_pieces_of_side('green')
        yield from self.get_pieces_of_side('orange')
        yield from self.get_pieces_of_side('white')

    def encode(self) -> str:
        return ''.join(self.get_pieces_in_order())


def filter_rectangles(squares: List[Square]) -> List[Square]:
    return [
        s for s in squares
        if 0.5 < s.height / s.width < 2
    ]


def filter_intersecting_squares(squares: List[Square]) -> List[Square]:
    ret = []
    for index, s in enumerate(squares):
        if is_last_from_intersection(s, squares[index + 1:]):
            ret.append(s)

    return ret


def get_square_color(full_image, square: Square) -> Optional[str]:
    avg_b = 0
    avg_g = 0
    avg_r = 0
    pixel_count = (
        (square.right - square.left - 6)
        * (square.bottom - square.top - 6)
    )

    for x in range(int(square.left) + 3, int(square.right) - 3):
        for y in range(int(square.top) + 3, int(square.bottom) - 3):
            pixel_color = full_image[y, x]
            avg_b += pixel_color[0]
            avg_g += pixel_color[1]
            avg_r += pixel_color[2]

    avg_color = np.uint8([[[
        avg_b / pixel_count, avg_g / pixel_count, avg_r / pixel_count
    ]]])
    # bgr to hsv
    avg_hsv = cv2.cvtColor(avg_color, cv2.COLOR_BGR2HSV)

    search = avg_hsv[0][0]
    for col_name, vals in hsv_colors.items():
        for low, high in vals:
            if (
                low[0] <= search[0] <= high[0]
                and low[1] <= search[1] <= high[1]
                and low[2] <= search[2] <= high[2]
            ):
                return col_name

    return None


def get_colors_by_index(
    squares: List[Square],
    colors: List[Optional[str]],
    dimension: int,
) -> IndexedColors:
    """Finds all colors on one side of the cube.
    First finds top left and bottom right cubes top left coordinates.
    Then approximates every squares coordinates and gets its color.
    """
    ret = {}
    min_left = squares[0].left
    min_top = squares[0].top
    max_left = squares[0].left
    max_top = squares[0].top
    for s in squares:
        if s.left < min_left:
            min_left = s.left
        if s.left > max_left:
            max_left = s.left
        if s.top < min_top:
            min_top = s.top
        if s.top > max_top:
            max_top = s.top

    max_left -= min_left
    max_top -= min_top
    per_square_left = round(max_left / (dimension - 1))
    per_square_top = round(max_top / (dimension - 1))
    for i, s in enumerate(squares):
        i_left = round((s.left - min_left) / per_square_left)
        i_top = round((s.top - min_top) / per_square_top)
        ret[(i_left, i_top)] = colors[i]

    return ret


def get_3x3_colors(
    indexed_colors: IndexedColors,
    dimension: int,
) -> KnownIndexedColors:
    ret: KnownIndexedColors = {}
    for j in range(0, dimension):
        for i in range(0, dimension):
            c = indexed_colors[(i, j)]
            if c is None:
                raise UnknownColorFound
            else:
                ret[(i, j)] = c
    return ret


def process_frame(cap: cv2.VideoCapture, state: CubeState) -> None:
    # get single frame from video
    ret, image = cap.read()
    if image is None:
        raise VideoDone

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    #  extract squares from image
    squares = [
        Square.normalize(s[0][0], s[0][1], s[2][0], s[2][1])
        for s in get_squares_from_frame(image)
    ]
    squares = filter_rectangles(squares)
    squares = filter_intersecting_squares(squares)

    # draw
    if state.all_squares_found(squares):
        line_color = (0, 255, 0)
        line_thickness = 4
    else:
        line_color = (255, 0, 255)
        line_thickness = 2

    for s in squares:
        cv2.rectangle(
            image,
            pt1=(s.left, s.bottom),
            pt2=(s.right, s.top),
            color=line_color,
            thickness=line_thickness
        )

    cv2.imshow("Image", image)
    k = cv2.waitKey(1) & 0xFF
    # press escape to exit
    if k == 27:
        raise KeyboardInterrupt

    colors = [
        get_square_color(blurred, c)
        for c in squares
    ]

    if not (len(squares) == state.squares_count == len(colors)):
        return

    indexed_colors = get_colors_by_index(squares, colors, state.dimension)

    try:
        indexed_colors_3x3 = get_3x3_colors(indexed_colors, state.dimension)
    except KeyError:
        return
    except UnknownColorFound:
        return

    state.add_side(indexed_colors_3x3)
    if state.all_sides_found():
        raise AllSidesFound


def main(video: Union[str, int]):
    cap = cv2.VideoCapture(video)
    state = CubeState()

    while True:
        try:
            process_frame(cap, state)
        except KeyboardInterrupt:
            print('User interrupt')
            break
        except VideoDone:
            break
        except AllSidesFound:
            print('All sides found')
            encoded_state = state.encode()
            solution = utils.solve(encoded_state, 'Kociemba')
            print(f'Solution: {solution}')
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video',
        default=0,
        help='Path to a video file.',
    )
    args = parser.parse_args()
    try:
        video = int(args.video)
    except ValueError:
        video = args.video

    main(video)
