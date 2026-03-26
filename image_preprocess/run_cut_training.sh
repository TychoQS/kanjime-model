MODEL_NAME="kanji_cut_128"
OUT_DIR="./output/cut_training/$MODEL_NAME"

if [ -d "$OUT_DIR" ]; then
    echo "ERROR: $OUT_DIR already exists."
    exit 1
fi
mkdir -p "$OUT_DIR" && \
conda run -n TFG-Cut python ./models/CUT/train.py \
  --dataroot ../data/unpaired_jpsc1400_etl9b \
  --checkpoints_dir ./output/cut_training \
  --name $MODEL_NAME \
  --CUT_mode CUT \
  --input_nc 3 \
  --output_nc 3 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --batch_size 2 \
  --load_size 128 \
  --crop_size 128 \
  --direction AtoB \
  --display_id 0 \
  2>&1 | tee ./output/cut_training/$MODEL_NAME/training_log.txt