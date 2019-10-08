import cv2
import numpy as np
import time
import imutils
from imutils import face_utils
import dlib
from abc import ABC, abstractmethod
from scipy.ndimage.interpolation import shift
from numba import jit
import pickle


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


class FaceDetector:
    def __init__(self, dat_file):
        self.dat_file = dat_file
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(self.dat_file)
        self.left_eye_idxs = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        self.right_eye_idxs = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        self.mouth_idxs = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

    @property
    def detector(self):
        return self._detector

    @property
    def predictor(self):
        return self._predictor


class Blink:
    start = 0
    start_ear = 1
    peak = 0
    peak_ear = 1
    end = 0
    end_ear = 0
    ear_of_foi = 0
    values = []
    blink_feature = np.zeros(shape=(4,), dtype=np.float32)

    def __init__(self):
        self.id = np.random.randint(0, 1e8)

    def summary(self):
        str = "ID: {}\n".format(self.id)
        str += "EAR Values:\n\tStart: {:.2f}\n\tPeak: {:.2f}\n\tEnd: {:.2f}\n".format(
            self.start_ear, self.peak_ear, self.end_ear)
        str += "Position Values:\n\tStart: {:.2f}\n\tPeak: {:.2f}\n\tEnd: {:.2f}\n".format(
            self.start, self.peak, self.end)
        str += "Amplitude: {:.3f}\n".format(self.amplitude)
        str += "Duration: {:.3f}\n".format(self.duration)
        str += "Velocity: {:.3f}\n\n".format(self.velocity)
        print(str)

    @property
    def amplitude(self):
        return (self.start_ear + self.end_ear - (2*self.peak_ear)) / 2.0

    @property
    def duration(self):
        return self.end-self.start+1

    @property
    def velocity(self):
        top = self.end_ear - self.peak_ear
        bottom = self.end - self.peak
        return top/bottom

    def is_not_noisy(self, amplitude_threshold):
        # peak_ear < start_ear and peak_ear < end  and amplitude>MIN_AMPLITUDE and start<peak:
        ear_position_check = self.start_ear > self.peak_ear < self.end_ear
        ear_amplitude_check = self.amplitude > amplitude_threshold
        frame_position_check = self.start < self.peak
        # print(self.start, self.peak)
        return ear_position_check and ear_amplitude_check and frame_position_check

    def is_not_imbalanced(self):
        a = (self.start_ear - self.peak_ear) > ((self.end_ear-self.peak_ear)*0.25)
        b = (self.end_ear - self.peak_ear) > ((self.start_ear - self.peak_ear)*0.25)
        return a and b


class BlinkDetector:
    _svm = None
    blink_counter = 0
    series = None
    reference_frame = np.uint64(0)
    skip = False
    frame_margin_btw_2blinks = 3
    min_amplitude = 0.04
    total_blinks = 0
    retrieved_blinks = []
    missed_blinks = False
    current_blink = Blink()
    last_blink = Blink()

    def __init__(self, path):
        self._svm = pickle.load(open(path, 'rb'))

    def predict_blink(self):
        return self._svm.predict(self.series.reshape(1, -1))

    def blink_detect_and_extract(self):
        assert self.series.any(), "Can not analyse blinks, if no ear series is present."

        # self.reference_frame continues on forever as long as program is running.
        # even at 32 bit, 30fps it would take over 4 years to cause an overflow
        # putting this here in case I ever reuse this code, and I am getting strange errors when
        # running for a long time.
        if self.reference_frame > 15:
            eyes_are_closed = self.predict_blink()
            # print(eyes_are_closed, self.reference_frame)
            if self.blink_counter == 0:
                self.current_blink = Blink()

            return self.blink_tracker(eyes_are_closed)
            # create function that:
            # retrieves blinks
            # keeps track of total blinks
            # updates general use counter
            # return when blink is ready

    def blink_tracker(self, eyes_closed):
        if self.blink_counter == 0:
            self.current_blink = Blink()
            self.current_blink.values = []

        blink_ready = False
        middle_of_series = self.series[6]

        # for brevity purposes
        cb = self.current_blink
        lb = self.last_blink

        # If the eyes are closed run this block of code
        if int(eyes_closed) == 1:
            cb.values.append(middle_of_series)
            cb.ear_of_foi = middle_of_series

            if self.blink_counter > 0:
                self.skip = False

            if self.blink_counter == 0:
                cb.start_ear = middle_of_series
                cb.start = self.reference_frame-6
            self.blink_counter += 1

            #print(cb.peak_ear, middle_of_series)
            if cb.peak_ear >= middle_of_series:
                cb.peak_ear = middle_of_series
                cb.peak = self.reference_frame-6

            # str = "\n\nCurrent Blink Attributes:\nValues: {}\n EAR of FOI: {}\nstart/start_ear: {}/{} peak/peak_ear: {}/{}".format(
            #     cb.values, cb.ear_of_foi, cb.start, cb.start_ear, cb.peak, cb.peak_ear)
        else:
            if self.blink_counter < 2 and self.skip == False:
                self.frame_margin_btw_2blinks = 8 if lb.duration > 15 else 1

                if((self.reference_frame-6) - lb.end) > self.frame_margin_btw_2blinks:
                    if lb.is_not_noisy(self.min_amplitude):
                        if lb.is_not_imbalanced():
                            blink_ready = True
                            # potential multiple blink event checked all passes, proceed...
                            # print(lb.values)
                            print("length of values: ", len(lb.values))
                            lb.values = np.convolve(
                                a=lb.values, v=[1/3, 1/3, 1/3], mode='same')  # smooth out signal

                            self.missed_blinks, self.retrieved_blinks = self.ultimate_grand_supreme_blink_detection(
                            )
                            self.total_blinks += len(self.retrieved_blinks)
                            self.blink_counter = 0
                            print(
                                "Detected {} blinks after post-processing".format(len(self.retrieved_blinks)))
                            return blink_ready
                        else:
                            self.skip = True
                            print("Rejected blink event due to imbalance")
                    else:
                        self.skip = True
                        print("Rejected due to noise, with a magnitude of: {}".format(
                            lb.amplitude))
                        #print(lb.start < lb.peak)

            if self.blink_counter > 1:
                # 7 ref points to the last frame that eyes were closed
                cb.end = self.reference_frame-7
                cb.end_ear = cb.ear_of_foi

                self.frame_margin_btw_2blinks = 8 if lb.duration > 15 else 1

                # merging two really close blinks
                if (cb.start-lb.end) <= self.frame_margin_btw_2blinks+1:
                    print("Merging")
                    frames_in_between = cb.start - lb.end-1
                    print("cb.start: {}, lb.end: {}".format(cb.start, lb.end-1))
                    print("s, e, fib: ", cb.start, lb.end, frames_in_between)
                    values_between = Utils.linear_interp(
                        start=lb.end_ear, end=cb.start_ear, n=frames_in_between)
                    print(lb.values, values_between, cb.values)
                    # lb.values = lb.values + values_between + cb.values
                    lb.values = np.concatenate(
                        (lb.values, values_between, cb.values)).tolist()
                    lb.end = cb.end
                    lb.end_ear = cb.end_ear

                    if lb.peak_ear > cb.peak_ear:
                        lb.peak_ear = cb.peak_ear
                        lb.peak = cb.peak

                else:
                    lb.values = cb.values
                    lb.end = cb.end
                    lb.end_ear = cb.end_ear

                    lb.peak = cb.peak
                    lb.peak_ear = cb.peak_ear

                    lb.start = cb.start
                    lb.start_ear = cb.start_ear

            self.blink_counter = 0
        self.retrieved_blinks = []

        return blink_ready

    def ultimate_grand_supreme_blink_detection(self):
        missed_blinks = False
        epsilon = 0.01
        y = np.array(self.last_blink.values)
        retrieved_blinks = []
        threshold = 0.4 * y.min() + 0.6*y.max()
        n = len(y)

        dy_dx = y[1:]-y[:-1]
        i = np.where(dy_dx == 0)[0]
        if len(i) != 0:
            for k in i:
                if k == 0:
                    dy_dx[0] -= epsilon
                else:
                    dy_dx[k] = epsilon * dy_dx[k-1]
        m = n-1
        c = dy_dx[1:m]*dy_dx[:m-1]
        x = np.where(c < 0)[0] + 1

        xtrema_ears = y[x]
        t = np.ones_like(x)
        t[xtrema_ears < threshold] = -1

        t = np.concatenate(([1], t, [1]))
        xtrema_ears = np.concatenate(([y[0]], xtrema_ears, [y[n-1]]))
        xtrema_idx = np.concatenate(([0], x, [n-1]))

        z = t[1:]*t[:-1]
        z_idx = np.where(z < 0)[0]
        num_of_blinks = len(z_idx) // 2

        ear1, ear2 = xtrema_ears[z_idx], xtrema_ears[z_idx+1]
        idx1, idx2 = xtrema_idx[z_idx], xtrema_idx[z_idx+1]

        missed_blinks = True if num_of_blinks > 1 else False
        if num_of_blinks == 0:
            print("no blinks detected")
            # print(t, self.last_blink.duration)
            # print(y)
            # print('Derivative: {}'.format(dy_dx))

        for j in range(num_of_blinks):
            blink = Blink()
            blink_values = np.hstack((ear1[2*j:2*j+2], ear2[2*j+1]))
            idx_values = np.hstack((idx1[2*j:2*j+2], idx2[2*j+1]))

            # print(blink_values)

            start_ear, peak_ear, end_ear = blink_values[0], blink_values[1], blink_values[2]
            start, peak, end = idx_values[0], idx_values[1], idx_values[2]

            blink.peak = peak
            blink.peak_ear = peak_ear

            blink.start = start
            blink.start_ear = start_ear

            blink.end = end
            blink.end_ear = end_ear
            retrieved_blinks.append(blink)

        return missed_blinks, retrieved_blinks

    @property
    def svm(self):
        return self._svm


class VideoImage:
    frame = None
    face_shape = None
    face_detector = None
    mar = None
    ear = None

    def __init__(self, frame=None):
        self.frame = frame

    def detect_face(self, face_detector):
        face_detected = False
        rects = face_detector.detector(self.frame, 0)
        if np.size(rects) != 0:
            face_detected = True
            self.face_shape = face_detector.predictor(self.frame, rects[0])
            self.face_shape = face_utils.shape_to_np(self.face_shape)
        else:
            self.putText("No face detected", pos=(10, 100))
        return face_detected

    def draw_mouth(self, idxs):
        assert self.face_shape.any(), "You have not called VideoImage.detect_face(), yet."

        m_start, m_end = idxs
        mouth = self.face_shape[m_start: m_end]
        self.mar = MouthRatio.calculate(mouth)
        mouth_hull = cv2.convexHull(mouth)
        self.putText("MAR: {0:.2f}".format(self.mar), pos=(400, 30))
        cv2.drawContours(self.frame, [mouth_hull], -1, (255, 0, 0), 1)

    def draw_eyes(self, l_idxs, r_idxs):
        assert self.face_shape.any(), "You have not called VideoImage.detect_face() yet."
        (l_start, l_end), (r_start, r_end) = l_idxs, r_idxs
        left_eye = self.face_shape[l_start:l_end]
        right_eye = self.face_shape[r_start:r_end]

        left_ear = EyeRatio.calculate(left_eye)
        right_ear = EyeRatio.calculate(right_eye)
        self.ear = (left_ear + right_ear) / 2.0
        hulls = [cv2.convexHull(eye) for eye in [left_eye, right_eye]]
        self.putText("EAR: {0:.2f}".format(self.ear), pos=(400, 50))
        cv2.drawContours(self.frame, [hulls[0]], -1, (0, 255, 0), 1)
        cv2.drawContours(self.frame, [hulls[1]], -1, (0, 255, 0), 1)

    def flip(self, axis=0):
        self.frame = cv2.flip(self.frame, axis)

    def putText(self, txt, pos=(10, 30)):
        cv2.putText(self.frame, txt, (pos[0], pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0, 2))

    def gray(self, keep_orig=False):
        if keep_orig:
            self.orig_frame = self.frame.copy()

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def display(self, title='Live Feed'):
        cv2.imshow(title, self.frame)

    def adjust_gamma(self, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) *
                          255 for i in np.arange(0, 256)]).astype("uint8")
        self.frame = cv2.LUT(self.frame, table)


class VideoCapture:
    _fps = None
    last_time = 0
    instance = None
    # required for usage in svm classifier for blink event, needs 13 frames.
    ear_series = np.zeros([13])
    # nothing like this has been implemented for usage as features in drowsiness detection.
    mar_series = None

    @property
    def fps(self):
        return self._fps

    def __init__(self, video_path):
        self.video_path = video_path

    def start(self, dat_file="./shape_predictor_68_face_landmarks.dat"):
        self.instance = cv2.VideoCapture(self.video_path)
        # self.instance.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        # self.instance.set(cv2.CAP_PROP_FRAME_WIDTH, 120)
        frame = VideoImage()
        fd = FaceDetector(dat_file=dat_file)
        bd = BlinkDetector(
            path="./Trained_SVM_C=1000_gamma=0.1_for 7kNegSample_0.21.3.sav")
        while True:
            self._calc_fps()
            ret, frame.frame = self.instance.read()
            bd.reference_frame += np.uint64(1)

            assert ret, "Could not read frame from video stream."

            frame.gray(keep_orig=False)
            frame.flip(axis=1)
            frame.putText("{0:.2f} fps".format(self.fps))
            frame.adjust_gamma(gamma=1.5)

            if frame.detect_face(fd):
                frame.draw_eyes(fd.left_eye_idxs, fd.right_eye_idxs)
                frame.draw_mouth(fd.mouth_idxs)
                self.ear_series = shift(self.ear_series, -1, cval=frame.ear)

                # update blink detectors ear series for analysis
                bd.series = self.ear_series

                # detect and extract blink features
                tmp = bd.total_blinks
                blink_ready = bd.blink_detect_and_extract()

                if blink_ready:
                    bd.reference_frame = 20
                    bd.skip = True
                    if tmp != bd.total_blinks:
                        print("total blinks: ", bd.total_blinks)

                    # need to reinit this attribute otherwise everything fucking breaks...
                    bd.last_blink.end = -10

            frame.display()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.instance.release()
        cv2.destroyAllWindows()

    @jit
    def _calc_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if self._fps is None:
            self._fps = 1.0 / dt
        else:
            s = np.clip(dt * 3, 0, 1)
            self._fps = self._fps * (1 - s) + (1.0 / dt) * s


if __name__ == '__main__':
    video = VideoCapture(0)
    video.start()
