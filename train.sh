source activate tf1.12
python training.py --run_name=BlitzNet300_instance_0310 --dataset=coco-seg --trunk=resnet50 --x4 --batch_size=8 --optimizer=adam --detect --instance --max_iterations=65000 --lr_decay 40000 50000 --instance_num 30
