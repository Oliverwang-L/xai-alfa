import numpy as np
import os
import pickle

from dsc.dsc_set import DSCSet
from tframe.utils import misc



class RMLSet(DSCSet):

  class Keys:
    raw_data = 'raw_data'
    SNRs = 'SNRs'

  SNR_LIST = np.linspace(-20, 18, 20).astype(int)
  CLASS_NAMES = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'GFSK', 'CPFSK',
                 'PAM4', 'WBFM', 'AM-SSB', 'AM-DSB']

  # region: Abstract Methods

  def configure(self, config_string: str):
    """config_string should be <snr_config>;<channels>
       <snr_config> should be
         (1)'*': load all data,
         (2) db1[,db2,...]: load signals which have a SNR of db1 (or db2 ...),
             e.g., '10,12'
         (3) db1-db2: load signals with SNR between db1 and db2
       <channels> should be an non-empty string consisting of non-repeat
         chars in ('i', 'q', 'a', 'p'), 'i'/'q' represent complex channels,
         'a' represents amplitude, and 'p' represents phase.
         e.g., 'iq', 'ap'
    """
    snr_config, channels = config_string.split(';')

    # Find SNRs to load by parsing config_string
    if snr_config in ('*', 'all'): snr_to_load = self.SNR_LIST
    elif '-' in snr_config:
      db_range = snr_config.split('-')
      min_db = min(self.SNR_LIST) if db_range[0] == '' else int(db_range[0])
      max_db = max(self.SNR_LIST) if db_range[1] == '' else int(db_range[1])
      snr_to_load = [snr for snr in self.SNR_LIST if min_db <= snr <= max_db]
    else: snr_to_load = [int(db) for db in snr_config.split(',')]

    # Check channels
    assert 0 < len(channels) < 5
    for c in channels: assert c in ('i', 'q', 'a', 'p')

    # TODO: ----------------

    # Set data_dict and SNRs
    data = self[self.Keys.raw_data]
    features, targets, SNRs = [], [], []
    for snr in snr_to_load:
      for i, modulation in enumerate(self.CLASS_NAMES):
        array = data[(modulation, snr)]
        features.append(array)
        targets.extend([i] * len(array))
        SNRs.extend([snr] * len(array))

    # Concatenate data
    self.features = np.swapaxes(np.concatenate(features, axis=0), 1, 2)
    self.targets = misc.convert_to_one_hot(targets, self.num_classes)
    self.properties[self.Keys.SNRs] = SNRs

    # Delete raw data in property
    self.properties.pop(self.Keys.raw_data)

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    # Load raw data as data_dict, with a format of each items:
    #   ('<class-name>', SNR): np.ndarray of shape (1000, 2, 128)
    raw_data = cls.load_raw_data(data_dir)
    data_set = RMLSet(name='RML2016.10a', n_to_one=True, raw_data=raw_data)
    data_set.properties[cls.NUM_CLASSES] = len(cls.CLASS_NAMES)
    return data_set

  @classmethod
  def load_raw_data(cls, data_dir):
    """Raw data should be placed at `52-DSC/data/rml2016` folder"""
    data_dir = os.path.join(data_dir, 'rml2016')
    file_name = 'RML2016.10a_dict.pkl'
    file_path = os.path.join(data_dir, file_name)
    # Make sure data file exists
    if not os.path.exists(file_path):
      raise FileExistsError(f'!! Cannot find `{file_name}` under `{data_dir}`')

    # Load raw data dict and return
    with open(file_path, 'rb') as f: return pickle.load(f, encoding='latin-1')

  def _check_data(self):
    """This method will be called during splitting dataset"""
    if self.Keys.raw_data in self.properties:
      assert len(self[self.Keys.raw_data]) == 220

  # endregion: Abstract Methods

  # region: Data Visualization

  def show(self):
    pass

  # endregion: Data Visualization



if __name__ == '__main__':
  pass


