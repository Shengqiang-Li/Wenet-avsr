""" Crop Face ROIs from videos for lipreading"""
import os
import cv2
import torch
import shutil
import tempfile
import argparse
import subprocess
import numpy as np

from tqdm import tqdm
from multiprocessing import Process
from torchvision import transforms as T


def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):                                                 
        ret, frame = cap.read() # BGR
        if ret:                      
            yield frame                                                    
        else:                                                              
            break                                                         
    cap.release()


def write_image(rois, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    image = rois[0]
    rois = rois[1:]
    for index, roi in enumerate(rois):
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
    parser.add_argument('--flist-path', help='list of file')
    parser.add_argument('--save-direc', default=None, help='the directory of saving face ROIs')
    parser.add_argument('--mp4-writer', default=None, help='choose ffmgpeg or cv2 to save face ROIs')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    # -- mouthROIs utils
    parser.add_argument('--roi-format', default='png', type=str, help='the format of mout ROIs')
    parser.add_argument('--crop-width', default=96, type=int, help='the width of face ROIs')
    parser.add_argument('--crop-height', default=96, type=int, help='the height of face ROIs')
    parser.add_argument('--resize-width', default=160, type=int, help='the width of resized image')
    parser.add_argument('--resize-height', default=160, type=int, help='the height of resized image')
    parser.add_argument('--nprocess', default=8, type=int, help='the number of process')


    args = parser.parse_args()
    return args


def align_face(fids, video_direc, save_direc, roi_format, mp4_writer, resize_width, resize_height, crop_width, crop_height, ffmpeg):

    for filename_idx, filename in enumerate(tqdm(fids)):

        video_pathname = os.path.join(video_direc, filename + '.mp4')

        dst_pathname = os.path.join(save_direc, filename + f'.{roi_format}')

        assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)

        if os.path.exists(dst_pathname):
            continue
        os.makedirs(os.path.dirname(dst_pathname), exist_ok=True)
        frame_gen = read_video(video_pathname)
        extract_transform = T.Compose([T.Resize((resize_width, resize_height)),
                                       T.CenterCrop((crop_width, crop_height)),]
        )
        frames = [extract_transform(torch.from_numpy(x).transpose(-2, -1).transpose(-3, -2)).transpose(-3, -2).transpose(-2, -1).numpy() for x in frame_gen]
        # -- save
        if roi_format == "mp4":
            if mp4_writer == "ffmpeg":
                write_video_ffmpeg(frames, dst_pathname)
            elif mp4_writer == "cv2":
                write_video_cv2(frames, dst_pathname)
        else:
            write_image(frames, dst_pathname)
    return


def split_files_list(inlist, chunksize):
    return [inlist[x:x + chunksize] for x in range(0, len(inlist), chunksize)]


def multiprocess_align_face(filename_path, video_direc, save_direc, roi_format, mp4_writer, resize_width, resize_height, crop_width, crop_height, ffmpeg, nprocess):
    files_list = [ln.strip() for ln in open(filename_path).readlines()]
    splitted_files_list = split_files_list(files_list, int((len(files_list) / nprocess)))

    process_list = list()
    for sub_files_list in splitted_files_list:
        print("len of sub_files_list", len(sub_files_list))
        p = Process(target=align_face, args=(sub_files_list, video_direc, save_direc, roi_format, mp4_writer, resize_width, resize_height, crop_width, crop_height, ffmpeg))
        process_list.append(p)
        p.Daemon = True
        p.start()
    for p in process_list:
        p.join()


if __name__ == '__main__':
    args = load_args()
    # -- mean face utils
    multiprocess_align_face(args.flist_path, args.video_direc, args.save_direc, 
                            args.roi_format, args.mp4_writer, args.resize_width, 
                            args.resize_height, args.crop_width, args.crop_height, 
                            args.ffmpeg, args.nprocess
                            )
    print('Done.')
