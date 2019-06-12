import cv2
import numpy as np
import os
import random
import time
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def create_folder(username):

    dir_test = 'saved/test/' + username
    dir_train = 'saved/train/' + username

    if not os.path.exists(dir_test):
        os.makedirs(dir_test)
        print("[INFO] Creating test directory: ", dir_test)

    if not os.path.exists(dir_train):
        os.makedirs(dir_train)
        print("[INFO] Creating train directory: ", dir_train)

# Choose between saving new user
# Or Recognizing existing user
# TODO: Make it run continuously
os.system("cls")
print("Face Recognition System:\n ")
print("What's your choice? ")
print("1- Create new user")
print("2- Recognize existing user")
print("3- Quit\n")
choice = int(input("> "))

# if user want to save new face
if choice == 1:
    # Open the webcam and take a picture on clicking space
    # The program will quit when clicking ESC
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Save New Face")
    img_counter = 0

    # Username
    username = input("Username: ")
    create_folder(username)
    try:
        os.makedirs(os.path.join("saved/train", username))
    except:
        pass
    new_path = os.path.join("saved", username)
    train_path = os.path.join(new_path, "train")
    user_path = os.path.join(train_path, username)

    i, j = 0, 0

    print("Turn your head in different positions and click space bar with every position")
    # TODO: Take pictures for 10 seconds to capture all the sides of the face
    num = 200
    while img_counter < num:
        ret, frame = cam.read()
        cv2.imshow("Save New Face", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.2, 5)

        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        # TODO: Make it take pictures automatically
        for x, y, w, h in face:

            # Get face in grayscale
            roi_gray = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), 1)
            gray_face = cv2.resize(roi_gray, (128, 128))

            # Capture only 20% of the face
            if i == 5:

                # Split data in train and test set to be 80%/20%
                if j % 5 == 0:
                    cv2.imwrite('saved/test/' + username + '/face_'
                                + username + '_' + str(j) + '.png', gray_face)
                else:
                    cv2.imwrite('saved/train/' + username + '/face_'
                                + username + '_' + str(j) + '.png', gray_face)

                j += 1
                i = 0
            i += 1

            os.system("cls")
            print(f"{img_counter+1}/{num} Face Saved!")
            time.sleep(0.1)
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    # train the model on those images
    #train_model(new_path)

elif choice == 2:
    # check for existing users in saved folder
    if len(os.listdir("saved/train")) == 0:
        print("There are no existing users.")
    # TODO: It only works with username "amine". FIX IT
    new_path = os.path.join("saved", "amine")
    test_path = os.path.join(new_path, "test")
    
    # train the model on those images
    classifier = "haarcascade_frontalface_default.xml"
    model = "models/model_face.h5"
    
    def get_label():
        label = os.listdir("saved/train")
        return label
    
    def get_legend(class_arg):
        label_list = get_label()
        # get label from prediction
        label = label_list[class_arg]
        # create color from each label
        coef = float(class_arg + 1)
        color = coef * np.asarray((20,30,50))
        return color, label
    
    def process_face(roi_gray):
        # resize input model size
        roi_gray = cv2.resize(roi_gray, (128, 128))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        return roi_gray
    
    # load haarcascade face classifier
    print("Loading cascade classifier...")
    face_cascade = cv2.CascadeClassifier(classifier)
    # Keras model was trained using the iPython Notebook
    print("Loading Keras model")
    model = load_model(model)

    """
    Open the webcam recognize the face.
    If face recognized print Access Granted.
    Else if face not recognized after 10 seconds Quit and Print
    Access not granted
    """
    # The program will quit when clicking ESC
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Recognizing Face...")

    # clear Terminal
    os.system('cls')
    time.sleep(2)
    print("Recognizing face...")
    img_counter = 0
    #prediction = 0
    # TODO: Change 120 to 10
    t_end = time.time() + 10
    # Run this loop for 10 seconds
    access = False
    while time.time() < t_end and access != True:
        ret, frame = cam.read()
        cv2.imshow("Testing existing Face", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        threshold = 0.5
        # Get faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_face = gray[y:y + h, x:x + w]
            roi_face = process_face(roi_face)
            prediction = model.predict(roi_face)
            print(prediction[0][0])            

            # Get label and color from prediction
            color, label = get_legend(np.argmax(prediction))

            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if(prediction[0][0] >= threshold):
                print(prediction[0][0])
                print("Access Granted. Welcome!")
                access = True
                break        
        
            
    if (time.time() > t_end and prediction[0][0] < threshold):
        print("Access Denied.\nFace Not Recognized")
    cam.release()
    cv2.destroyAllWindows()

    
    

    
elif choice == 3:
    print("Quitting...")

else:
    print("Wrong choice.") # TODO: get user input until correct choice