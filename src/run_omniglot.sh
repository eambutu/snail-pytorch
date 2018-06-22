python train.py --exp omniglot_5way_1shot --cuda >> omniglot_5way_1shot.txt
python train.py --exp omniglot_5way_5shot --num_samples 5 --cuda >> omniglot_5way_5shot.txt
python train.py --exp omniglot_20way_1shot --num_cls 20 --cuda >> omniglot_20way_1shot.txt
python train.py --exp omniglot_20way_5shot --num_cls 20 --num_samples 5 --cuda >> omniglot_20way_5shot.txt
