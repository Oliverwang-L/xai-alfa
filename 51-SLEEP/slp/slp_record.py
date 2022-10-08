import numpy as np
import pickle

from tframe.data.augment.img_aug import image_augmentation_processor
from tframe import DataSet
from typing import Tuple
from tframe.utils import misc
from tframe import console
import os



class SleepRecord(object):

  @classmethod
  def load(cls):
    from slp_core import th

    return cls.load_sleepdata()

  @classmethod
  def load_sleepdata(cls):
    from slp_core import th

    from slp.slp_agent import SLPAgent
    from slp.slp_datasets.sleepedfx import SleepEDFx

    # Find 51-SLEEP/data/sleepedfx

    data_name = th.data_config.split(':')[0]
    data_num = th.data_config.split(':')[1]
    suffix = '-alpha'
    suffix_num = '' if data_num is None else f'({data_num})'
    tfd_preprocess_path = os.path.join(
      th.data_dir, data_name, f'{data_name}{suffix_num}{suffix}-final.tfds')
    if os.path.exists(tfd_preprocess_path):
      with open(tfd_preprocess_path,'rb') as input_:
        console.show_status('Loading `{}` ...'.format(tfd_preprocess_path))
        dataset = pickle.load(input_)
    else:
      dataset: SleepEDFx = SLPAgent.load_as_tframe_data(data_dir=th.data_dir,
                                                        data_name=data_name,
                                                        first_k=data_num,
                                                        suffix=suffix)

      dataset = dataset.partition_slp(rnn=th.use_rnn)

      with open(tfd_preprocess_path, 'wb') as output_:
        console.show_status(f'Saving {tfd_preprocess_path}...')
        pickle.dump(dataset, output_, pickle.HIGHEST_PROTOCOL)
      console.show_status('Finishing split dataset to (train_set, val_set, test_set)...')

    train_set, val_set, test_set = dataset

    return train_set, val_set, test_set


