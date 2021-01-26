# -Top-Related-Meta-Learning-Method-for-Few-Shot-Detection
cvpr2021 code
loss.py: our contribution(TCL-C and category-based grouping mechanism)

# train base model, such as novel id:2
python2 train_meta.py cfg/metayolo.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg cfg/fegment.cfg ../weight/darknet19_448.conv.23

# metayolo.data
metayolo=1

metain_type=2

data=voc

neg = 1

rand = 0

novel = data/voc_novels.txt

novelid = 2

scale = 1

meta = data/voc_traindict_full.txt

train = /home1/liqian/data/voc/voc_train.txt

valid = /home1/liqian/data/voc/2007_test.txt

backup = coco/ours

gpus=0,1,2,3


# tune on novel classes
python2 train_meta.py cfg/metatune_2shot.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg cfg/fegment.cfg backup/metayolo_novel2_neg1_metayolo/000350.weights

# cfg/metatune_2shot.data
metayolo=1

metain_type=2

data=voc

tuning = 1

neg = 0

rand = 0

novel = data/voc_novels.txt

novelid = 2

max_epoch = 4000

repeat = 200

dynamic = 0

scale=1

train = /home1/liqian/data/voc/voc_train.txt

meta = data/voc_traindict_bbox_2shot.txt

valid = /home1/liqian/data/voc/2007_test.txt

backup = backup/metatunetest1

gpus=0,1,2,3


# test
python2 valid_ensemble.py cfg/metatune_2shot.data cfg/darknet_dynamic.cfg cfg/reweighting_net.cfg cfg/fegment.cfg backup/metatunetest1_novel2_neg0_metatune_2shot/ 1

# eval
python2 scripts/voc_eval.py results/metatunetest1_novel2_neg0_metatune_2shot_test/





