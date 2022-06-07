from dsc.dsc_set import DSCSet
from dsc.dsc_agent import DSCAgent



def load():
  from dsc_core import th

  train_set, val_set, test_set = DSCAgent.load(
    th.data_dir, th.val_proportion, th.test_proportion)

  assert isinstance(train_set, DSCSet)
  assert isinstance(val_set, DSCSet)
  assert isinstance(test_set, DSCSet)

  # input_shape and output_dim should be determined here
  th.input_shape = train_set.features[0].shape
  th.output_dim = train_set.num_classes

  return train_set, val_set, test_set



if __name__ == '__main__':
  from dsc_core import th

  th.data_config = 'rml:*'
  train_set, val_set, test_set = load()
  print()