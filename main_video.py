import cv2
from simple_facerec import SimpleFacerec
import tkinter as tk
import face_recognition
import os

# Encode faces from a folder
sfr = SimpleFacerec()
# sfr.load_encoding_images("images/")
sfr.load_saved_encoding()
# Create root for name input bar
# root = tk.Tk()
# Load Camera
cap = cv2.VideoCapture(0)
frameLast = ""
ButtonClicked = False
name = ""
def MyClick():
    global ButtonClicked, root, frame, e, name
    name = e.get()
    root.destroy()
    path = "images/" + name + '.png'
    cv2.imwrite(path, frame)
    sfr.encode_a_face(path, name)
    # cv2.imwrite(path, frame)
    # sfr.load_encoding_images(path)
    
def Detect(frame):

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    # print(face_locations, face_names)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.destroyAllWindows()
    cv2.imshow("Frame", frame)

    key = cv2.waitKey()
    cv2.destroyAllWindows()


def DetectFace(frame, draw = True):
    # rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(frame)
    if draw == False:
        return frame
    if len(face_location) != 1:
        return frame
    for (top, right, bot, left) in face_location:
        cv2.rectangle(frame, (left, top), (right, bot), (255, 0, 0))
        frame = frame[top : bot, left: right]
    return frame
    # return frame

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == 27:
        Detect(frame)
    if key == ord('q'):
        exit(0)
    if key == ord(' '):
        root = tk.Tk()
        e = tk.Entry(root)
        e.pack()
        myButton = tk.Button(root, text = "Submit my name", command=MyClick)
        myButton.pack()
        
        root.mainloop()
    if key == ord('1'):
        sfr.encode_a_face("images/Minh le")
        # cv2.imwrite("")

