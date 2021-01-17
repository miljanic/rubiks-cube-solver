#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 miljanic <miljanic.m@protonmail.com>
#
# Distributed under terms of the BSD 3-Clause license.
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List

import imutils
import numpy as np
import cv2


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# Point = Tuple[float, float]
class Point(NamedTuple):
    x: float
    y: float


@dataclass
class Square:
    a: Point
    b: Point
    c: Point
    d: Point

    def is_nested(self, s: 'Square'):
        return self.a.x < s.a.x and self.a.y < s.a.y and self.c.x > s.c.x and self.c.y > s.c.y


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


cap = cv2.VideoCapture("kocka.mp4")

while (True):
    ret, image = cap.read()

    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    cv2.rectangle(image, pt1=(200, 200), pt2=(300, 300), color=(0, 0, 255), thickness=10)
    img = cv2.GaussianBlur(image, (5, 5), 0)
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
    # filter nested squares

    real_squares = [
        Square(
            Point(s[0][0], s[0][1]),
            Point(s[1][0], s[1][1]),
            Point(s[2][0], s[2][1]),
            Point(s[3][0], s[3][1])
        )
        for s in squares
    ]


    def is_biggest(big: Square, squares: List[Square]):
        for small in squares:
            if small.is_nested(big):
                return False
        return True


    for_drawing = [
        s for s in real_squares if is_biggest(s, real_squares)
    ]

    print(for_drawing)

    for c in for_drawing:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        # M = cv2.moments(c)
        # cX = int((M["m10"] / M["m00"]) * ratio)
        # cY = int((M["m01"] / M["m00"]) * ratio)
        # shape = sd.detect(c)
        # # multiply the contour (x, y)-coordinates by the resize ratio,
        # # then draw the contours and the name of the shape on the image
        # c = c.astype("float")
        # c *= ratio
        # c = c.astype("int")
        # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

        # cv2.rectangle(image, pt1=(c[0][0], c[0][1]), pt2=(c[2][0], c[2][1]), color=(0, 0, 255), thickness=1)
        cv2.rectangle(image, pt1=c.a, pt2=c.c, color=(0, 0, 255), thickness=1)
        # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), 2)
        # show the output image
    # print('arstarst', squares[0])
    squares = []
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    # Capture frame-by-frame
    # ret, frame = cap.read()
    # ret, img = cap.read()
    #
    # img = cv2.resize(img, (500, 500))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # for cnt in contours:
    #     if cv2.contourArea(cnt) > 5000:  # remove small areas like noise etc
    #         hull = cv2.convexHull(cnt)  # find the convex hull of contour
    #         hull = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
    #         if len(hull) == 4:
    #             cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # img = cv2.resize(img, (500, 500))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # for cnt in contours:
    #     if cv2.contourArea(cnt) > 5000:  # remove small areas like noise etc
    #         hull = cv2.convexHull(cnt)  # find the convex hull of contour
    #         hull = cv2.approxPolyDP(hull, 0.1 * cv2.arcLength(hull, True), True)
    #         if len(hull) == 4:
    #             cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
    #     cv2.imshow('frame', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # ret, thresh_img = cv2.threshold(blur, 91, 255, cv2.THRESH_BINARY)
    #
    # contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #
    # for c in contours:
    #     cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
