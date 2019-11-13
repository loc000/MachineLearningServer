wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P data/
wget http://jaina.cs.ucdavis.edu/datasets/adv/captioning/attention-model-best.tar.bz2 -P data/
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz -P data/
wget http://jaina.cs.ucdavis.edu/datasets/adv/captioning/word_to_idx.pkl -P data/train/

tar xvf data/inception_v3_2016_08_28.tar.gz -C data/
tar xvf data/attention-model-best.tar.bz2
