import cv2
from simple_facerec import SimpleFacerec
import tkinter as tk


# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
# Create root for name input bar
# root = tk.Tk()
# Load Camera
cap = cv2.VideoCapture(0)
frameLast = ""
ButtonClicked = False
def MyClick():
    global ButtonClicked, root, frame, e
    name = e.get()
    path = "images/" + name + ".png"
    cv2.imwrite(path, frame)
    sfr.load_encoding_images(path)
    root.destroy()

while True:
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == 27:
        frameLast = frame
        break
    if key == ord('q'):
        exit(0)
    if key == ord(' '):
        root = tk.Tk()
        e = tk.Entry(root)
        e.pack()
        myButton = tk.Button(root, text = "Submit my name", command=MyClick)
        myButton.pack()
        
        root.mainloop()
        # cv2.imwrite("")

    
cap.release()
frame = frameLast

# Detect Faces
face_locations, face_names = sfr.detect_known_faces(frame)
for face_loc, name in zip(face_locations, face_names):
    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

    cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

cv2.destroyAllWindows()
cv2.imshow("Frame", frame)

key = cv2.waitKey()
cv2.destroyAllWindows()