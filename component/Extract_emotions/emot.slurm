#!/bin/bash
#SBATCH --partition=donut-default
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50GB

python3.6 scripts/test.py --model-path=2021-09-09-18\:42\:30_checkpoint.pt --test-path=emotion_text_NEWS.csv --output-path=emotion_text_NEWS_filled.csv
python3.6 scripts/test.py --model-path=2021-09-09-18\:42\:30_checkpoint.pt --test-path=emotion_text_Telegram.csv --output-path=emotion_text_Telegram_filled.csv

### TO TRAIN:
##python3.6 scripts/train.py --train-path data/train/2018-E-c-En-train.txt --dev-path data/dev/2018-E-c-En-dev.txt

### TO TEST
### python3.6 scripts/test.py --model-path=2021-09-09-18\:42\:30\_checkpoint.pt --test-path=test_tweet.txt
####MF_telegram_updated_w2v_emot.txt
