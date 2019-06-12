# Face Recognition

This is a Tensorflow implementation of a Real-Time Face recognition system using Opencv.

## Plan
**V1.0**
- [X] Setup the environment and the Github Repo
- [X] Access the webcam using Opencv
- [X] Add the ability to save a new face and the name of the user
- [X] Use Tensorflow to recognize saved face and print Access Granted
- [X] Write the ReadMe documentation

## Requirements
- Keras
- OpenCv

## How To
**Step 1: Create necessary folders**

Start by creating 2 folders: 
"saved" for the training and testing set.
"models" for the trained model.
Inside the "saved" folder create 2 folders: "train" and "test".

**Step 2: Add new user**

Run the command: 

```python face_recognition.py```

Choose first choice to create a new user. The program will take your name as input then it will capture 200 pictures of your face and split them 80% for training and 20% for testing.
In "saved/train" and "saved/test" you'll find 2 new folders with the username you wrote. Add a new folder in "train" and "test" with any name you want, and put some random people images inside these folders from Google (example 8 in "train" and 2 in "test").
Then zip the "saved" folder to get: saved.zip

**Step 3: Create the training model**

Because I don't have an Nvidia GPU, I created my model using Google Colab.
Start by running the Jupyter Notebook "face_training.ipynb". Then upload saved.zip and run all the cells of the notebook.
When you finish, you will have a new file called "model_face.h5" download it and save it to the models folder locally.

**Step 4: Run the program**

Now you have all the files. You can run: "python face_recognition.py" and test it with you webcam or add new users. 
If the user is recognized it will output: "Access Granted!".
Else if the user is not recognized after 10 seconds of trying it will output: "Access Denied!".
