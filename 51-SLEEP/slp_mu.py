from tframe import Classifier
from tframe import mu

from slp_core import th
from tframe.models import Recurrent
from tframe.layers import Input, Linear, Activation, Rescale

from tframe.layers.hyper.dense import Dense
from tframe.configs.config_base import Config



def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  if flatten: model.add(mu.Flatten())
  return model


def finalize(model):
  from tframe import context

  assert isinstance(model, Classifier)
  model.add(mu.Dense(th.output_dim, use_bias=False, prune_frac=0.5))
  model.add(mu.Activation('softmax'))

  # Build model
  # context.customized_loss_f_net = add_customized_loss_f_net
  model.build(batch_metric=['accuracy'])
  return model


def typical(cells):
  assert isinstance(th, Config)
  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)
  # Add layers
  model.add(Input(sample_shape=th.input_shape))
  # Add hidden layers
  if not isinstance(cells, (list, tuple)): cells = [cells]
  for cell in cells: model.add(cell)
  # Build model and return
  output_and_build(model)
  return model


def output_and_build(model):
  assert isinstance(model, Classifier)
  assert isinstance(th, Config)
  # Add output layer
  model.add(Dense(num_neurons=th.output_dim))
  model.add(Activation('softmax'))

  model.build(metric='accuracy', batch_metric='accuracy', last_only=True)
