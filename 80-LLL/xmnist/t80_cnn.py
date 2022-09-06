import lll_core as core
import lll_mu as m

from tframe.enums import SaveMode
from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cnn'
id = 9
def model():
  th = core.th
  model = m.get_container(flatten=False)

  for i, c in enumerate(core.th.archi_string.split('-')):
    if c == 'p':
      model.add(m.mu.MaxPool2D(pool_size=2, strides=2))
      continue

    c = int(c)
    model.add(m.mu.Conv2D(
      filters=c, kernel_size=th.kernel_size, use_bias=False,
      activation=th.activation, use_batchnorm=th.use_batchnorm and i > 0))
    if th.dropout > 0: model.add(m.mu.Dropout(1 - th.dropout))

  # Add flatten layer
  model.add(m.mu.Flatten())
  return m.finalize(model)


def main(_):
  console.start('{} on LLL task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.task = th.Tasks.FMNIST
  th.input_shape = [28, 28, 1]
  th.output_dim = 10

  th.data_config = 'beta:0.8'
  # th.data_config = 'alpha:2,1,1,1'
  th.train_id = 0
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '64-p-32'
  th.kernel_size = 3
  th.activation = 'relu'
  th.use_batchnorm = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100000
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.003
  th.patience = 5

  th.validation_per_round = 2

  th.train = True
  th.overwrite = True if th.train_id == 0 else False
  th.save_mode = SaveMode.ON_RECORD
  th.print_cycle = 10

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # LLL setups
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  th.cl_reg_config = 'moses'
  th.cl_reg_lambda = 1.0

  th.export_tensors_to_note = True
  th.developer_code = ''
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

