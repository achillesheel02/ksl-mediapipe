import numpy as np
import glob
import utils
from keras.models import load_model
from collections import deque
import cv2
from tqdm import tqdm
import ffmpeg

TEST_VIDEO_PATH = 'test_videos/'
test_videos_paths = glob.glob(TEST_VIDEO_PATH + '*.*')
MODEL_PATH = 'models/model.h5'
model = load_model(MODEL_PATH)
classes = utils.generate_labels().classes_
SEQ_LENGTH = utils.get_sequence_length()

for path in test_videos_paths:
    print('Processing...')
    data = utils.convert_to_csv(path)
    filename = path.split('/')[-1].split('.')[0]
    target_video = TEST_VIDEO_PATH + filename + '_annotated.mp4'
    cap = cv2.VideoCapture(target_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(TEST_VIDEO_PATH + filename + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          30,
                          (frame_width, frame_height))
    counter = 0
    frames = deque(maxlen=SEQ_LENGTH)
    frame_length = data.shape[0]

    while counter < frame_length:
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        frames.append(data[counter, :])
        counter += 1
        outputs = []
        output_name = ''

        if len(frames) is SEQ_LENGTH:
            input_data = np.array(frames)
            input_data = input_data[np.newaxis, np.newaxis, ...]
            output = model.predict(input_data)
            output_name = classes[np.argmax(output)]
            output_value = output[0][np.argmax(output)]
            print( output_name , output_value)
            if output_value > 0.85:
                outputs.append([output_name, output_value])
                cv2.putText(frame, 'Prediction: ' + output_name, (50, 50), font, 1, (0, 255, 255), 1, cv2.LINE_4)

        out.write(frame)

    out.release()
    ffmpeg.input(TEST_VIDEO_PATH + filename + '.avi').output(TEST_VIDEO_PATH + filename + '_compressed.mp4').run()
    cap.release()
    cv2.destroyAllWindows()

    print('Video saved\n\n')
