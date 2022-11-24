import os

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 tsv preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--video_dir', type=str, help='directory contains mouth roi')
    parser.add_argument('--video_format', type=str, help='format of video file')
    parser.add_argument('--tgt_dir', type=str, help='target directory to save processed data')
    parser.add_argument('--file_list', type=str, help='file list')
    parser.add_argument('--label_list', type=str, help='label list')
    parser.add_argument('--valid-ids', type=str, help='a list of valid ids')
    args = parser.parse_args()
    file_list, label_list = f"{args.lrs3}/{args.file_list}", f"{args.lrs3}/{args.label_list}"
    assert os.path.isfile(file_list) , f"{file_list} not exist -> run lrs3_prepare.py first"
    assert os.path.isfile(label_list) , f"{label_list} not exist -> run lrs3_prepare.py first"

    audio_dir, video_dir = f"{args.lrs3}/audio", f"{args.lrs3}/{args.video_dir}"

    def setup_target(target_dir, train, valid, test):
        for name, data in zip(['train', 'dev', 'test'], [train, valid, test]):
            dir_name = os.path.abspath(f"{target_dir}/{name}")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with open(os.path.join(dir_name, "wav.scp"), 'w') as fout:
                for fid, _ in data:
                    _, prefix, postfix = fid.split("/")
                    file_id = f"{prefix}/{postfix}"
                    fout.write('\t'.join([file_id, os.path.abspath(f"{audio_dir}/{fid}.wav")])+'\n')
            with open(os.path.join(dir_name, "text"), 'w') as fout:
                for fid, label in data:
                    _, prefix, postfix = fid.split("/")
                    file_id = f"{prefix}/{postfix}"
                    fout.write('\t'.join([file_id, label])+'\n')
            with open(os.path.join(dir_name, "video.scp"), 'w') as fout:
                for fid, _ in data:
                    _, prefix, postfix = fid.split("/")
                    file_id = f"{prefix}/{postfix}"
                    fout.write('\t'.join([file_id, os.path.abspath(f"{video_dir}/{fid}.{args.video_format}")])+'\n')
        return

    fids, labels = [x.strip() for x in open(file_list).readlines()], [x.strip().upper() for x in open(label_list).readlines()]
    valid_fids = set([x.strip() for x in open(args.valid_ids).readlines()])
    train_all, train_sub, valid, test = [], [], [], []
    for fid, label in zip(fids, labels):
        part = fid.split('/')[0]
        if part == 'test':
            test.append([fid, label])
        else:
            if fid in valid_fids:
                valid.append([fid, label])
            else:
                train_all.append([fid, label])
                if part == 'trainval':
                    train_sub.append([fid, label])

    dir_433h = args.tgt_dir
    print(f"Set up 433h dir")
    os.makedirs(dir_433h, exist_ok=True)
    setup_target(dir_433h, train_all, valid, test)
    return


if __name__ == '__main__':
    main()
