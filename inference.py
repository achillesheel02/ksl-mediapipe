import numpy as np
import glob
import utils
from keras.models import load_model
from collections import deque

TEST_VIDEO_PATH = 'test_videos/'
test_videos_paths = glob.glob(TEST_VIDEO_PATH + '*.mov')
MODEL_PATH = 'models/model_low_loss.h5'
model = load_model(MODEL_PATH)
classes = utils.generate_labels().classes_
SEQ_LENGTH = utils.get_sequence_length()

for path in test_videos_paths:
    print('Processing...')
    data = utils.convert_to_csv(path)
    filename = path.split('/')[-1].split('.')[0]
    print(filename)
    counter = 0
    frames = deque(maxlen=SEQ_LENGTH)
    frame_length = data.shape[0]


    while counter < frame_length:
        frames.append(data[counter, :])
        counter += 1
        outputs = []

        if len(frames) is SEQ_LENGTH:
            input_data = np.array(frames)
            input_data = input_data[np.newaxis, ...]
            output = model.predict(input_data)
            output_name = classes[np.argmax(output)]
            output_value = output[0][np.argmax(output)]
            if output_value > 0.5:
                outputs.append([output_name, output_value])

        print("Frame " + str(counter) + ": ", outputs)


    print('Done with video\n\n')