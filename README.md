# IMGCaptioning

Computer Vision - Class Project

For the details please refer to the report.

## Authors

- Dayou Du <dayoudu@nyu.edu>
- Xin Chen <xc1113@nyu.edu>

## Prerequisities

- Python >= 3.6
- Pytorch >= 0.4
- pycocotools

## Usage

1. Download and unpack MSCOCO 2017 dataset(train/val)
2. Build the vocabulary with `build_vocab.py --caption_path=<train caption json file> --vocab_path=<vocab saving path> --threshold=<minimum word count threshold>`
3. Modify the training set and validation set paths in `main.py`
4. Do the first phase training (where the CNN part is frozen) with `main.py cnn_model=<base cnn model> --num_epochs=<number of epochs to train> --leanring_rate=<learning rate> --hidden_size=<LSTM network output/hidden size> --train_cnn=False --saving_model_path=<path to store the check-points>`
5. To continue the training process from a checkpoint, please use `--encoder_model_path` and `--decoder_model_path` options. 
6. Do the second phase training (where the CNN part is unlocked), simply set `--train_cnn=True`
7. By default (for the speed consideration), the beam search is disabled. To enable it simply set `--beam_width` option (a `beam_with` larger than 1 will automatically trigger the beam search during inference.

## Acknowledgement

- The COCO dataloaders are modified based on [PyTorch's COCO Caption dataset loader](https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py)
- The BLEU score calculator are modifed based on Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>'s work.
- The models are trained on NYU's Prince cluster

