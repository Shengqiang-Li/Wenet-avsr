import os
import cv2
import dlib
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames

def detect_landmark(image, cnn_detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = cnn_detector(gray)
    rects = [d.rect for d in rects]
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def detect_face_landmarks(face_predictor_path, cnn_detector_path, root_dir, landmark_dir, fids):
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor = dlib.shape_predictor(face_predictor_path)
    input_dir = root_dir #
    output_dir = landmark_dir #
    for fid in tqdm(fids):
        video_path = os.path.join(f"{input_dir}/lipread_mp4", fid+'.mp4')
        output_fn = os.path.join(f"{output_dir}", fid+'.pkl')
        frames = load_video(video_path)
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, cnn_detector, predictor)
            landmarks.append(landmark)
        if os.path.exists(output_fn):
            continue
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, 'wb'))
    return

def split_files_list(inlist, chunksize):
    return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]

def multiprocess_detect_face_landmarks(face_predictor_path, cnn_detector_path, root_dir, landmark_dir, flist_dir, split, nprocess):
    flist_fn = f'{flist_dir}/{split}.txt'
    files_list = [ln.strip() for ln in open(flist_fn).readlines()]
    print(len(files_list))
    splitted_files_list = split_files_list(files_list, int((len(files_list) / nprocess)))

    process_list = list()
    for sub_files_list in splitted_files_list:
        print("len of sub_files_list", len(sub_files_list))
        p = Process(target=detect_face_landmarks, args=(face_predictor_path, cnn_detector_path, root_dir, landmark_dir, sub_files_list))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='detecting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='root dir')
    parser.add_argument('--landmark', type=str, help='landmark dir')
    parser.add_argument('--manifest', type=str, help='a directory contains file list')
    parser.add_argument('--split', type=str, help='split of dataset')
    parser.add_argument('--cnn_detector', type=str, help='path to cnn detector (download and unzip from: http://dlib.net/files/mmod_human_face_detector.dat.bz2)')
    parser.add_argument('--face_predictor', type=str, help='path to face predictor (download and unzip from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)')
    parser.add_argument('--nprocess', type=int, help='number of process')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    args = parser.parse_args()
    import skvideo
    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    print(skvideo.getFFmpegPath())
    import skvideo.io
    multiprocess_detect_face_landmarks(args.face_predictor, args.cnn_detector, args.root, args.landmark, args.manifest, args.split, args.nprocess)
