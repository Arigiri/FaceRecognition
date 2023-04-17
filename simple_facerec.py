import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_saved_encoding(self):
        path = os.path.join("images", "EncodingFiles")
        images_path = glob.glob(os.path.join(path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))
        for encode_path in images_path:
            encode = np.load(encode_path)
            (filename, ext) = os.path.splitext(os.path.basename(encode_path))
            self.known_face_encodings.append(encode)
            self.known_face_names.append(filename)
    
    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))
        path = os.path.join("images", "EncodingFiles")
        try:
            os.mkdir(path)
        except:
            pass
        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            print(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_location = face_recognition.face_locations(rgb_img, 2)
            img_encoding = face_recognition.face_encodings(rgb_img, img_location)[0]
            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
            np.save(os.path.join(path, filename + ".npy"), img_encoding)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, 2)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            print(matches)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
    
    def encode_a_face(self, path_to_face, name):
        face = cv2.imread(path_to_face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_location = face_recognition.face_locations(face, 2)
        path = os.path.join("images", "EncodingFiles")
        try:
            face_encode = face_recognition.face_encodings(face, face_location)[0]
            self.known_face_encodings.append(face_encode)
            self.known_face_names.append(name)
            np.save(os.path.join(path, name + ".npy"), face_encode)
        except:
            print("Cannot detect any face")
