# Offtarget-Predict-palnt-model

python finetune_nt_pytorch_multifeature.py     --train_csv data/train.csv     --dev_csv data/dev.csv     --test_csv data/test.csv     --batch_size 8     --epochs 80     --lr 2e-5 --device cuda:0 --fp16 --max_length 64 --freeze_backbone --ckpt_dir pt
