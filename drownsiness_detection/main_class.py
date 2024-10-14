import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils


# class structure
class DetectDrownsiness:
    """This class contains all functions and required usage
    This can be activated by calling run program"""

    # initial

    def __init__(self, cam_number: int = 0, threshold: float = 0.2):
        self.threshold = threshold
        self.cam_num = cam_number
        self.flag = 0  # reset every starting of the main loop

        # load dlib features
        self.detect = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # face landmarks
        (self.lstart, self.lend) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rstart, self.rend) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        # camera
        self.cam = cv2.VideoCapture(self.cam_num)

    @staticmethod
    def calculate_ear(eye):
        """This function returns eye aspect ratio"""
        height1 = distance.euclidean(eye[1], eye[5])
        height2 = distance.euclidean(eye[2], eye[4])
        lenght = distance.euclidean(eye[0], eye[3])

        ear = (height1 + height2) / (2.0 * lenght)

        return ear

    def run_the_program(self):
        while True:
            ret, frame = self.cam.read()

            if ret:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                subjects = self.detect(gray, 0)

                for subject in subjects:

                    # predict
                    shape = self.predictor(gray, subject)
                    shape = face_utils.shape_to_np(shape)

                    # eyes
                    left_eye = shape[self.lstart:self.lend]
                    right_eye = shape[self.rstart:self.rend]

                    # calcualte ears
                    right_ear = self.calculate_ear(right_eye)
                    left_ear = self.calculate_ear(left_eye)

                    # total ear ratio
                    ear = (left_ear + right_ear) / 2.0

                    if ear < self.threshold:
                        self.flag += 1
                        if self.flag > 20:

                            # send signal
                            print("*" * 10 + "ALERT" + "*" * 10)
                            self.flag = 0
            else:
                break

c = DetectDrownsiness()
c.run_the_program()