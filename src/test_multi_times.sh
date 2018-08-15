#! /bin/zsh
workon gpu-py3-chainer
for i in {1..10}
do
    python train_han.py
done
