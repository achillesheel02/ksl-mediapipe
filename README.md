### **Kenyan Sign Language Recognition System**
#### Technologies Used
1.  MediaPipe 
2.  Tensorflow
3.  Python
4. FFMpeg
5.  OpenCV

Trained using LSTM model

## How to Initialize
1.  Clone repo.
2.  Install [MediaPipe](https://github.com/google/mediapipe).
3.  Move `mediapipe` folder into root directory of repo.
4.  Replace `simple_run_graph_main.cc` in `mediapipe/mediapipe/examples/desktop/` with the one in the repo.
5.  In your terminal, in the root directory of the repo, run the following to build binaries: 
`bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu`


## Preparing your Dataset
1.  Your labels are going to be populated in the `labels.txt` file, one per line.
2.  Create a `test_data`, `test_data_annotated`, `models`, `test_videos` and `csv_outputs` folder at root.
3.  Save video samples in the`test_data` directory, each label having its own folder, i.e. `test_data/eat/eat_1.mp4`. `.mkv` and `.webm` files and video files without a header throw an error when passed through Mediapipe. Label names in `labels.txt` MUST correspond to their directory names.

## Training
1. Create a virtual Python 3 environment.
2. Run `pip install -r requirements.txt`.
3. Change the `SEQ_LENGTH` variable in`utils.py` as desired. (How many frames per video to be used for training). Default is 50 (~2 secconds). This implies that the script will only use the first 50 frames of every clip in the dataset.
3.  By default, the CNN Time Distrbitued model is used. To use a different model, change `line 78` of `train_sequential.py`. Upon changing you have to prepare the data by reshaping it accordingly for the chosen model's input. Reshaping functions are in the `load_data()` function in the script.
3. Run `train_sequential.py`.
4. Models will be saved in `models` directory. 3 are saved: After training, lowest loss and highest accuracy.

## Inference
1.  Inference is done on videos saved in the `test_videos` directory.
2.  Choose model to use in `line 12` on `inference.py`.
4. If another model is used, you have to reshape the frame accordingly on `line 42` of `inference.py`.
4. Run `inference.py`
2.  Final version is a `.avi` file. `ffmpeg` is used to create a smaller `.mp4` version. Other files created in between are the annotated video and the `.csv` of landmarks. 
3. Predictions are annotated on the videos.
