#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_CLIPstyler.py --content_path ./test_set/face.jpg \
--content_name face --exp_name exp_picasso \
--text "Picasso" \
--source_text "a Photo"
