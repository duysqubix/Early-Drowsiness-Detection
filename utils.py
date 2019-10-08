import numpy as np
from abc import ABC, abstractmethod


class Utils:

    @staticmethod
    def linear_interp(start, end, n):
        m = (end-start)/(n+1)
        x = np.linspace(1, n, n)
        y = m*(x-0)+start
        return y


class AspectRatio(ABC):

    @staticmethod
    @abstractmethod
    def calculate(shape):
        pass

    @staticmethod
    def euclidean(a, b):
        return np.linalg.norm(a - b)


class MouthRatio(AspectRatio):

    @staticmethod
    # @jit
    def calculate(shape):
        mouth = shape
        # A = dist.euclidean(mouth[14], mouth[18])
        # C = dist.euclidean(mouth[12], mouth[16])
        a = AspectRatio.euclidean(mouth[14], mouth[18])
        c = AspectRatio.euclidean(mouth[12], mouth[16])
        mar = 0.2 if c < 0.1 else a / c
        return mar


class EyeRatio(AspectRatio):

    @staticmethod
    # @jit
    def calculate(shape):
        eye = shape

        a = AspectRatio.euclidean(eye[1], eye[5])
        b = AspectRatio.euclidean(eye[2], eye[4])
        c = AspectRatio.euclidean(eye[0], eye[3])
        ear = 0.3 if c < 0.1 else ((a + b) / (2.0 * c))
        if ear >= 0.45:
            ear = 0.45

        return ear
