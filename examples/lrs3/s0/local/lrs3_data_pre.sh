lrs3=/mnt/mnt-data-2/shengqiang.li/corpus/lrs3
ffmpeg=/usr/bin/ffmpeg
nshard=4
stage=6
stop_stage=6

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "preparing data..."
  step=1
  if  [ ${stage} != 4 ]; then
    python lrs3_prepare.py \
      --lrs3 ${lrs3} \
      --ffmpeg $ffmpeg \
      --rank ${rank} \
      --nshard ${nshard} \
      --step ${step}
  else
    python lrs3_prepare.py \
      --lrs3 ${lrs3} \
      --step ${step}
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Detect facial landmark"
  cnn_detector=/mnt/mnt-data-2/shengqiang.li/pre_modules/mmod_human_face_detector.dat
  face_detector=/mnt/mnt-data-2/shengqiang.li/pre_modules/shape_predictor_68_face_landmarks.dat
  manifest=file.list
  for rank in $(seq 0 $((nshard - 1)));do
    echo "current shard is $rank"
    python local/data_preparation/detect_landmark.py \
      --root ${lrs3} \
      --landmark ${lrs3}/landmark \
      --manifest ${lrs3}/$manifest \
      --cnn_detector $cnn_detector \
      --face_detector $face_detector \
      --ffmpeg $ffmpeg \
      --nshard ${nshard} \
      --rank ${rank}
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Crop mouth ROIs"
  landmark_dir=landmark
  save_dir=image_jpeg
  file_name=file256.list
  mean_face=/mnt/mnt-data-2/shengqiang.li/pre_modules/20words_mean_face.npy
  roi_format=jpeg
  nprocess=8
  mp4_writer=cv2
  python local/data_preparation/align_mouth.py \
    --video-direc ${lrs3} \
    --landmark ${lrs3}/${landmark_dir} \
    --flist-path ${lrs3}/${file_name} \
    --save-direc ${lrs3}/${save_dir} \
    --mean-face ${mean_face} \
    --roi-format ${roi_format} \
    --nprocess ${nprocess} \
    --ffmpeg $ffmpeg \
    --mp4-writer $mp4_writer
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Crop face ROIs"
  landmark_dir=landmark
  save_dir=image_face
  file_name=file_sp.list
  roi_format=png
  nprocess=8
  mp4_writer=cv2
  python local/data_preparation/align_face.py \
    --video-direc ${lrs3} \
    --flist-path ${lrs3}/${file_name} \
    --save-direc ${lrs3}/${save_dir} \
    --roi-format ${roi_format} \
    --nprocess ${nprocess} \
    --ffmpeg $ffmpeg \
    --mp4-writer $mp4_writer
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Generate manifest"
  valid_data=/mnt/mnt-data-2/shengqiang.li/corpus/lrs3/lrs3-valid.id
  file_list=file.list
  label_list=label.list
  tgt_dir=/mnt/mnt-data-2/shengqiang.li/code/wenet_os/examples/lrs3/s0/data_avsp_face_new
  video_dir=image_face
  video_format=png
  python local/data_preparation/lrs3_manifest.py \
    --lrs3 ${lrs3} \
    --tgt_dir ${tgt_dir} \
    --valid-ids $valid_data \
    --file_list $file_list \
    --label_list $label_list \
    --video_dir $video_dir \
    --video_format $video_format
fi