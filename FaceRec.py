import cv2
import face_recognition
import os


home_dir = os.path.dirname(__file__)

#Face Cascade
face_cascade = cv2.CascadeClassifier(os.path.join(home_dir,'opencv-files/haarcascade_frontalface_alt.xml'))
eye_left_cascade = cv2.CascadeClassifier(os.path.join(home_dir, 'opencv-files/haarcascade_mcs_lefteye_alt.xml'))
eye_right_cascade = cv2.CascadeClassifier(os.path.join(home_dir, 'opencv-files/haarcascade_mcs_righteye_alt.xml'))
mouth_cascade = cv2.CascadeClassifier(os.path.join(home_dir, 'opencv-files/haarcascade_mcs_mouth.xml'))
nose_cascade = cv2.CascadeClassifier(os.path.join(home_dir, 'opencv-files/haarcascade_mcs_nose.xml'))

# Load a sample picture and learn how to recognize it.
akhil_image = face_recognition.load_image_file(os.path.join(home_dir,"faces/akhil.JPG"))
akhil_face_encoding = face_recognition.face_encodings(akhil_image)[0]
sheeba_image = face_recognition.load_image_file(os.path.join(home_dir,"faces/sheeba.JPG"))
sheeba_face_encoding = face_recognition.face_encodings(sheeba_image)[0]
amit_image = face_recognition.load_image_file(os.path.join(home_dir,"faces/amit.JPG"))
amit_face_encoding = face_recognition.face_encodings(amit_image)[0]
pravin_image = face_recognition.load_image_file(os.path.join(home_dir,"faces/Pravin.jpg"))
pravin_face_encoding = face_recognition.face_encodings(pravin_image)[0]
sreeni_image = face_recognition.load_image_file(os.path.join(home_dir,"faces/Sreeni.jpg"))
sreeni_face_encoding = face_recognition.face_encodings(sreeni_image)[0]


known_faces = [ akhil_face_encoding, sheeba_face_encoding, amit_face_encoding, pravin_face_encoding, sreeni_face_encoding ]

def FaceRecognition(frame):
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True


    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            global known_faces
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            if match[0]:
                name = "Akhil"
            elif match[1]:
                name = "Sheeba"
            elif match[2]:
                name = "Amit"
            elif match[3]:
                name = "Pravin"
            elif match[4]:
                name = "Sreeni"

            return name

def face_detect(img, path):
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    rect = None
    face = None
    for (x,y,w,h) in faces:
        #rect = (x, y, w, h)
        #face = img[y:y+h, x:x+w]
        #img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(path+"\\face.png", img[y:y+h, x:x+w])
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes_left = eye_left_cascade.detectMultiScale(roi_gray)
        eyes_right = eye_right_cascade.detectMultiScale(roi_gray)
        nose = nose_cascade.detectMultiScale(roi_gray, 1.7, 11)
        lips = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
        for (ex,ey,ew,eh) in eyes_left:
            #cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
            cv2.imwrite(path+"\\left_eye.png", roi_color[ey:ey+eh, ex:ex+ew])
            #break
        for (rx,ry,rw,rh) in eyes_right:
            #cv2.rectangle(roi_color, (rx,ry), (rx+rw, ry+rh), (0,255,0),2)
            cv2.imwrite(path+"\\right_eye.png", roi_color[ry:ry+rh, rx:rx+rw])
            #break
        for (lx,ly,lw,lh) in lips:
            #cv2.rectangle(roi_color, (lx,ly), (lx+lw, ly+ int(0.75*lh)), (0,0,255),2)
            cv2.imwrite(path+"\\lips.png", roi_color[ly:ly+int(0.75*lh), lx:lx+lw])
            #break
        for (nx,ny,nw,nh) in nose:
            #cv2.rectangle(roi_color, (lx,ly), (lx+lw, ly+ int(0.75*lh)), (0,0,255),2)
            cv2.imwrite(path+"\\nose.png", roi_color[ny:ny+int(0.75*nh), nx:nx+nw])
            #break
    #os.system("scp output\face.png \\D-113108704\PrcessedImage")

    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return faces, rect

