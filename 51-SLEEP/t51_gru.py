from tframe import tf
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.gru import GRU
from tframe.nets.rnn_cells.lstms import LSTM
from tframe.utils.organizer.task_tools import update_job_dir
from slp_core import th

import slp_core as core
import slp_mu as m


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'sleep-gru'
id = 4
def model():
  assert isinstance(th, m.Config)
  cells = []

  cells.append(m.mu.Dense(500))

  cell = LSTM(
    state_size=th.state_size,
    use_reset_gate=th.use_reset_gate,
  )
  cells.append(cell)
  return m.typical(cells)


def main(_):
  console.start('{} on TO task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.sequence_length = 5
  th.input_shape = [3000]
  th.data_config = 'sleepedf:10'
  th.output_dim = 5
  th.use_rnn = True
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setu
  # ---------------------------------------------------------------------------
  th.model = model
  th.state_size = 100
  th.use_reset_gate = True

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  # th.max_iterations = 50000
  th.epoch = 1000
  th.batch_size = 32

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  th.train = True
  th.overwrite = True
  th.print_cycle = 10
  th.save_model = True
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.export_tensors_upon_validation = True

  # th.note_cycle = 300
  # th.validate_cycle = 10
  # th.export_states = True
  # th.export_gates = True
  # th.export_dl_ds_stat = True
  # th.export_dl_dx = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '{}'.format(th.sequence_length)
  th.mark = '{}({})'.format(model_name, th.state_size) + tail
  th.gather_summ_name = th.prefix + summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
