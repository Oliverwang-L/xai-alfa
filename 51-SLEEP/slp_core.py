import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from slp.slp_config import SLPConfig as Hub
# from tframe import DefaultHub as Hub
from tframe import Classifier

import slp_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.50

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
# th.input_shape = [120, 2]
th.input_shape = [3000, 3]

th.early_stop = True
th.patience = 20

th.print_cycle = 10
th.validation_per_round = 2

th.export_tensors_upon_validation = True

th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True

th.val_batch_size = 100
th.eval_batch_size = 100

#per sample length
th.random_sample_length = 3000

def activate():
  # Load data
  train_set, val_set, test_set = du.load_data()
  if th.centralize_data: th.data_mean = train_set.feature_mean

  # Build model
  assert callable(th.model)
  model = th.model()

  assert isinstance(model, Classifier)

  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, test_set=test_set,
                trainer_hub=th)
  else:
    # Evaluate on test set
    model.evaluate_pro(test_set, batch_size=100, verbose=True, cell_width=4,
                       show_confusion_matrix=True, show_class_detail=True)

  # End
  model.shutdown()
  console.end()
