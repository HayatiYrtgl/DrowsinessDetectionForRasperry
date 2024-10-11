import dlib
import imutils
import cv2
from scipy.spatial import distance
from imutils import face_utils


def calculate_ear(eye):
    """This function returns eye aspect ratio"""
    height1 = distance.euclidean(eye[1], eye[5])
    height2 = distance.euclidean(eye[2], eye[4])
    lenght = distance.euclidean(eye[0], eye[3])

    ear = (height1 + height2) / (2.0*lenght)

    return ear

# load dlib features
detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# face landmarks
(lstart, lend) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# open the camera
cam = cv2.VideoCapture(0)

# process the frame
while 1:
    ret, frame = cam.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:

            # predict
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            # eyes
            left_eye = shape[lstart:lend]
            right_eye = shape[rstart:rend]

            # calcualte ears
            right_ear = calculate_ear(right_eye)
            left_ear = calculate_ear(left_eye)






