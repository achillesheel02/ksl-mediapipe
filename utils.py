import csv
import glob
import pandas as pd
import numpy as np
import subprocess
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
import math


def calculateDistance(x2, y2):
    dist = math.sqrt((x2 - 0) ** 2 + (y2 - 0) ** 2)
    return dist


# paths
CSV_PATH = 'csv_outputs/'
TEST_DATA_PATH = 'test_data/'
TEST_DATA_ANNOTATED_PATH = 'test_data_annotated/'

SEQ_LENGTH = 48


def get_sequence_length():
    return SEQ_LENGTH


def generate_labels():
    LABELS = []
    with open('labels.txt', 'r') as file:
        for line in file:
            LABELS.append(line.split('\n')[0])

    lb = LabelBinarizer()
    LABELS = np.array(LABELS)
    lb.fit(LABELS)
    return lb


def execute_mediapipe_csv():
    path_to_mediapipe_binary = "bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_tflite"
    graph = "mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop.pbtxt"

    training_videos = glob.glob(TEST_DATA_PATH + '*/*.mov')
    annotated_videos = glob.glob(TEST_DATA_ANNOTATED_PATH + '*/*.mp4')
    annotated_videos = [item.replace(TEST_DATA_ANNOTATED_PATH, '').replace('mp4', '') for item in annotated_videos]
    training_videos = [item for item in training_videos if
                       item.replace(TEST_DATA_PATH, '').replace('mov', '') not in annotated_videos]
    for video in tqdm(training_videos, desc='Converting mediapipe files to csv'):
        ANNOTATED_ENTRIES = [entry.name for entry in os.scandir(TEST_DATA_ANNOTATED_PATH) if entry.is_dir()]
        CSV_ENTRIES = [entry.name for entry in os.scandir(CSV_PATH) if entry.is_dir()]
        abspath = '/Users/bachillah/Documents/Python/KSL/'
        input_video_path = os.path.join(abspath, video)

        label = video.split('/')[-2]
        path_for_annotation = TEST_DATA_ANNOTATED_PATH + label
        path_for_csv = CSV_PATH + label

        if label not in ANNOTATED_ENTRIES:
            os.mkdir(path_for_annotation)

        if label not in CSV_ENTRIES:
            os.mkdir(path_for_csv)

        video_save_dir = video.split('/')[-1].split('.')[0]
        output_video_path = path_for_annotation + '/' + video_save_dir + '.mp4'
        output_video_path = os.path.join(abspath, output_video_path)
        output_csv_path = path_for_csv + '/' + video_save_dir + '.csv'
        output_csv_path = os.path.join(abspath, output_csv_path)

        cmd = [

            path_to_mediapipe_binary,
            "--calculator_graph_config_file=%s" % graph,
            "--input_side_packets=input_video_path=%s,output_video_path=%s" % (input_video_path, output_video_path),
            "--output_stream=multi_hand_landmarks",
            "--output_stream_file=%s" % output_csv_path
        ]
        subprocess.run(cmd, capture_output=True, cwd="mediapipe")

def convert_to_csv(video):

    path_to_mediapipe_binary = "bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_tflite"
    graph = "mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop.pbtxt"

    abspath = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(abspath, video)
    path_for_annotation = TEST_DATA_ANNOTATED_PATH + 'test_videos'
    path_for_csv = CSV_PATH + 'test_videos'

    video_save_dir = video.split('/')[-1].split('.')[0]
    output_video_path = path_for_annotation + '/' + video_save_dir + '.mp4'
    output_video_path = os.path.join(abspath, output_video_path)
    output_csv_path = path_for_csv + '/' + video_save_dir + '.csv'
    output_csv_path = os.path.join(abspath, output_csv_path)

    if not os.path.exists(output_csv_path):
        cmd = [
            path_to_mediapipe_binary,
            "--calculator_graph_config_file=%s" % graph,
            "--input_side_packets=input_video_path=%s,output_video_path=%s" % (input_video_path, output_video_path),
            "--output_stream=multi_hand_landmarks",
            "--output_stream_file=%s" % output_csv_path
        ]
        print(subprocess.run(cmd, capture_output=True, cwd="mediapipe"))
    df = pd.read_csv(output_csv_path, sep="#", names=['TimeStamp', 'Landmarks'])
    extracted_landmarks_col = []
    for row in df['Landmarks'].dropna():
        extracted_landmarks = extract_landmarks(row)
        extracted_landmarks_col.append(extracted_landmarks)

    extracted_landmarks_col = np.array(extracted_landmarks_col)
    return extracted_landmarks_col

def add_padding(sequence):
    sequence_length = sequence.shape[0]

    if sequence_length < SEQ_LENGTH:
        padding = np.zeros((SEQ_LENGTH - sequence_length, 42))
        return np.append(sequence, padding, axis=0)

    elif sequence_length == SEQ_LENGTH:
        return sequence

    elif sequence_length > SEQ_LENGTH:
        return sequence[:SEQ_LENGTH, :]


def extract_landmarks(landmarks_string):
    landmarks_list = landmarks_string.split(" ")
    final_landmarks = []
    for landmark in landmarks_list:
        if landmark in ['', 'NaN']:
            continue
        landmark = landmark[landmark.find("(") + 1:landmark.find(")")]
        landmarkXY = landmark.split(",")
        distance_from_origin = calculateDistance(float(landmarkXY[0]), float(landmarkXY[1]))
        final_landmarks.append(distance_from_origin)

    while len(final_landmarks) < 42:
        if len(final_landmarks) < 42:
            final_landmarks.append(0)

    return np.array(final_landmarks)


def create_dataset():
    dataset = []
    labels = []
    csv_items = glob.glob(CSV_PATH + '*/*.csv')
    for file in tqdm(csv_items, desc='Creating training dataset'):
        df = pd.read_csv(file, sep="#", names=['TimeStamp', 'Landmarks'])

        label = file.split('/')[1]
        extracted_landmarks_col = []
        for row in df['Landmarks'].dropna():
            extracted_landmarks = extract_landmarks(row)
            extracted_landmarks_col.append(extracted_landmarks)

        extracted_landmarks_col = np.array(extracted_landmarks_col)
        extracted_landmarks_col = np.array(add_padding(extracted_landmarks_col))
        labels.append(label)
        dataset.append(extracted_landmarks_col)

    return np.array(dataset), np.array(labels)

