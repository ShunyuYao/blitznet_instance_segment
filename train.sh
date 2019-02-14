source activate tensorflow
python training.py --run_name=BlitzNet300_dishes --dataset=coco-seg --trunk=resnet50 --x4 --batch_size=16 --optimizer=adam --detect --segment --max_iterations=65000 --lr_decay 40000 50000 --ckpt=11000
