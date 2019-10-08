import numpy as np
import imutils
from imutils import face_utils
import cv2
from numba import jit
import time
from scipy.ndimage.interpolation import shift
from utils import MouthRatio, EyeRatio
from detector import FaceDetector, BlinkDetector


class VideoImage:
    frame = None
    face_shape = None
    face_detector = None
    mar = None
    ear = None
    cum_number_of_frames = np.uint64(0)

    def __init__(self, frame=None):
        self.frame = frame

    def detect_face(self, face_detector):
        face_detected = False
        rects = face_detector.detector(self.frame, 0)
        if np.size(rects) != 0:
            face_detected = True
            self.cum_number_of_frames += 1
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
                    blink_freq = bd.total_blinks / frame.cum_number_of_frames

                    for detected_blink in bd.retrieved_blinks:
                        detected_blink.summary()

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
