""" Crop Face ROIs from videos for lipreading"""
import os
import cv2
import pickle
import shutil
import tempfile
import argparse
import subprocess
import numpy as np

from tqdm import tqdm
from collections import deque
from skimage import transform as tf
from multiprocessing import Process


# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped


def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):                                                 
        ret, frame = cap.read() # BGR
        if ret:                      
            yield frame                                                    
        else:                                                              
            break                                                         
    cap.release()


# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def write_image(rois, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    image = rois[0]
    rois = rois[1:]
    for i_roi, roi in enumerate(rois):
        image = np.concatenate((image, roi), axis=0)
    cv2.imwrite(target_path, image)
    return


def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return

def write_video_cv2(rois, target_path, audio_path=None):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    sil_mp4_path = os.path.join(tmp_dir, "sil.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        vw = cv2.VideoWriter(str(sil_mp4_path), fourcc, fps, (96, 96))
        for roi in rois:
            vw.write(roi)
    except Exception as e:
        raise e
    finally:
        vw.release()
    if audio_path is None:
        shutil.copy2(sil_mp4_path, target_path)
    shutil.rmtree(tmp_dir)
    return



def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--landmark-direc', default=None, help='landmark directory')
    parser.add_argument('--flist-path', help='list of file')
    parser.add_argument('--save-direc', default=None, help='the directory of saving mouth ROIs')
    parser.add_argument('--mp4-writer', default=None, help='whether use ffmgpeg or cv2 to save mouth ROIs')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    # -- mean face utils
    parser.add_argument('--mean-face', type=str, help='reference mean face (download from: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy)')
    # -- mouthROIs utils
    parser.add_argument('--roi-format', default='png', type=str, help='the format of mout ROIs')
    parser.add_argument('--crop-width', default=96, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=96, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start-idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed_landmarks')
    parser.add_argument('--nprocess', default=8, type=int, help='the number of process')


    args = parser.parse_args()
    return args


def crop_patch(video_pathname, landmarks, mean_face_landmarks, stablePntsIDs, std_size, window_margin, start_idx, stop_idx, crop_height, crop_width):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           std_size)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        crop_height//2,
                                        crop_width//2,))
        if frame_idx == len(landmarks)-1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, std_size)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[start_idx:stop_idx],
                                            crop_height//2,
                                            crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def align_mouth(fids, video_direc, landmark_direc, save_direc, roi_format, mp4_writer, crop_width, crop_height, window_margin, start_idx, stop_idx, ffmpeg, std_size, mean_face_landmarks, stablePntsIDs):

    for filename_idx, filename in enumerate(tqdm(fids)):

        video_pathname = os.path.join(video_direc, filename + '.mp4')

        landmarks_pathname = os.path.join(landmark_direc, filename+'.pkl')

        dst_pathname = os.path.join(save_direc, filename + f'.{roi_format}')

        assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)
        assert os.path.isfile(landmarks_pathname), "File does not exist. Path input: {}".format(landmarks_pathname)

        if os.path.exists(dst_pathname):
            continue

        landmarks = pickle.load(open(landmarks_pathname, 'rb'))

        # -- pre-process landmarks: interpolate frames not being detected.
        preprocessed_landmarks = landmarks_interpolate(landmarks)

        if not preprocessed_landmarks:
            print(f"resizing {filename}")
            frame_gen = read_video(video_pathname)
            frames = [cv2.resize(x, (crop_width, crop_height)) for x in frame_gen]
            if roi_format == "mp4":
                write_video_ffmpeg(frames, dst_pathname)
            else:
                write_image(frames, dst_pathname)
            continue

        # -- crop
        sequence = crop_patch(video_pathname, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, std_size, window_margin=args.window_margin, start_idx=args.start_idx, stop_idx=args.stop_idx, crop_height=args.crop_height, crop_width=args.crop_width)
        assert sequence is not None, "cannot crop from {}.".format(filename)

        # -- save
        os.makedirs(os.path.dirname(dst_pathname), exist_ok=True)
        if roi_format == "mp4":
            if mp4_writer == "ffmpeg":
                write_video_ffmpeg(sequence, dst_pathname)
            elif mp4_writer == "cv2":
                write_video_cv2(sequence, dst_pathname)
        else:
            write_image(sequence, dst_pathname)
    return


def split_files_list(inlist, chunksize):
    return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]


def multiprocess_align_mouth(filename_path, video_direc, landmark_direc, save_direc, roi_format, mp4_writer, crop_width, crop_height, window_margin, start_idx, stop_idx, ffmpeg, nprocess, std_size, mean_face_landmarks, stablePntsIDs):
    files_list = [ln.strip() for ln in open(filename_path).readlines()]
    splitted_files_list = split_files_list(files_list, int((len(files_list) / nprocess)))

    process_list = list()
    for sub_files_list in splitted_files_list:
        print("len of sub_files_list", len(sub_files_list))
        p = Process(target=align_mouth, args=(sub_files_list, video_direc, landmark_direc, save_direc, roi_format, mp4_writer, crop_width, crop_height, window_margin, start_idx, stop_idx, ffmpeg, std_size, mean_face_landmarks, stablePntsIDs))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()

if __name__ == '__main__':
    args = load_args()
    # -- mean face utils
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(args.mean_face)
    stablePntsIDs = [33, 36, 39, 42, 45]
    multiprocess_align_mouth(args.flist_path, args.video_direc, args.landmark_direc, 
                             args.save_direc, args.roi_format, args.mp4_writer, args.crop_width, 
                             args.crop_height, args.window_margin, args.start_idx, 
                             args.stop_idx, args.ffmpeg, args.nprocess,
                             STD_SIZE, mean_face_landmarks, stablePntsIDs
                             )
    print('Done.')
