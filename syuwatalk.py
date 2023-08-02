import threading
import cv2
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os
from matplotlib import pyplot as plt    
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import scipy
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import pyttsx3
from queue import Queue
from PIL import ImageFont, ImageDraw, Image
from keras.models import load_model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam






    
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
tts_engine = pyttsx3.init()
tts_engine = pyttsx3.init()

# Set voice (optional)
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[0].id)
new_rate = 150
tts_engine.setProperty('rate', new_rate)

# Queue to store recognized actions



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image = cv2.flip(image, 1)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results   

def draw_landmarks(image, results):
    mp_drawing.drawing_utils.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
def draw_styled_landmarks(image, results):
# Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                         ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
def test_tracking():
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            # Draw landmarks
            draw_styled_landmarks(image, results)
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
    
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['ありがとう','お疲れ様','ごめんなさい','また会おう','嬉しい','怒る','悲しい','無言'])

# Thirty videos worth of data
no_sequences = 20

# Videos are going to be 30 frames in length
sequence_length = 50

def create ():
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def take_key():
    cap = cv2.VideoCapture(0)
# Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

def take_data ():
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)  
    
    return X, y



colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def tts_speak():
    while True:
        
        action = recognized_actions_queue.get()
        if action and action != "無言":
            tts_engine.say(action)
            sentence.clear()
            tts_engine.runAndWait()
            action=None
            time.sleep(0.5)
        elif action== "無言":
            sentence.clear()
            tts_engine.runAndWait()
            action=None
            time.sleep(0.5)
        
        

        # Empty the queue after speaking the action
        while not recognized_actions_queue.empty():
            recognized_actions_queue.get_nowait()
            recognized_actions_queue.task_done()
        
        


recognized_actions_queue = Queue()
        
def draw_japanese_text(image, text, position):
    fontpath = 'C:/Windows/Fonts/YuGothL.ttc'
    font = ImageFont.truetype(fontpath, 35)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Draw black box as the background
    text_width, text_height = draw.textsize(text, font=font)
    box_left, box_top = position
    box_right = box_left + text_width
    box_bottom = box_top + text_height
    draw.rectangle([box_left, box_top, box_right, box_bottom], fill='black')

    # Draw Japanese text on top of the black box
    draw.text(position, text, font=font, fill='white')
    image = np.asarray(pil_image)
    return image


def user(model):
    global sentence
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.6
    last_action_time = time.time()

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        tts_thread = threading.Thread(target=tts_speak)
        tts_thread.daemon = True
        tts_thread.start()

        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-50:]
            
            if len(sequence) == 50:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Add recognized action to the queue
                recognized_action = ' '.join(sentence)
                recognized_actions_queue.put(recognized_action)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Draw the Japanese text and the black box at the top
            text_to_display = ' '.join(sentence)
          #  cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)  # Black box
            image_with_text = draw_japanese_text(image.copy(), ' '.join(sentence), (10, 40))

            cv2.imshow('OpenCV Feed',image_with_text)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



def alternative_model():
    model_path = 'model_alternative.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        X, y = take_data()

        # Data normalization
        X_normalized = X / X.max()  # You can use other normalization techniques if needed

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.05)

        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)

        optimizer = Adam(lr=0.001)

        # Input layer
        input_layer = Input(shape=(50, 1662))

        # 1D Convolutional layers
        conv_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        conv_layer = MaxPooling1D(pool_size=2)(conv_layer)
        conv_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(conv_layer)
        conv_layer = MaxPooling1D(pool_size=2)(conv_layer)

        # LSTM layers
        lstm_layer = LSTM(64, return_sequences=True, activation='relu')(conv_layer)
        lstm_layer = LSTM(128, return_sequences=True, activation='relu')(lstm_layer)
        lstm_layer = LSTM(64, return_sequences=False, activation='relu')(lstm_layer)

        # Fully connected layers
        fc_layer = Dense(64, activation='relu')(lstm_layer)
        fc_layer = Dropout(0.2)(fc_layer)
        fc_layer = Dense(32, activation='relu')(fc_layer)
        fc_layer = Dropout(0.2)(fc_layer)

        # Output layer
        output_layer = Dense(actions.shape[0], activation='softmax')(fc_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=700, callbacks=[tb_callback])

        model.summary()
        model.save('model_alternative.h5')

    return model


model=alternative_model()             
user(model)