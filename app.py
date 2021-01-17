#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 miljanic <miljanic.m@protonmail.com>
#
# Distributed under terms of the BSD 3-Clause license.


import numpy as np
import cv2

cap = cv2.VideoCapture("kocka.mp4")

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh_img = cv2.threshold(blur, 91, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    for c in contours:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
