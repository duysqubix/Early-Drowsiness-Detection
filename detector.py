import numpy as np
import dlib
from imutils import face_utils
import pickle
from utils import Utils


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
                            #print("length of values: ", len(lb.values))
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
                    #print("cb.start: {}, lb.end: {}".format(cb.start, lb.end-1))
                    #print("s, e, fib: ", cb.start, lb.end, frames_in_between)
                    values_between = Utils.linear_interp(
                        start=lb.end_ear, end=cb.start_ear, n=frames_in_between)
                    #print(lb.values, values_between, cb.values)
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
        self.retrieved_blinks = 0

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
