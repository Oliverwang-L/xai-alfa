import os, sys

from pictor import Pictor
from slp_record import SleepRecord
from tframe.data.augment.img_aug import image_augmentation_processor


def load_data():
  # Load data
  # ...
  from slp_core import th

  train_set, val_set, test_set = SleepRecord.load()
  if th.augmentation:
    train_set.batch_preprocessor = image_augmentation_processor
  return train_set, val_set, test_set



if __name__ == '__main__':

  train_set, val_set, test_set = load_data()

  # Initiate a pictor
  # p = pictor(title='sleep monitor', figure_size=(15, 9))

  # set plotter
  # m = monitor()
  # p.add_plotter(m)

  # set objects
  # p.objects = sleep_data_list

  # Begin main loop
  # p.show()
