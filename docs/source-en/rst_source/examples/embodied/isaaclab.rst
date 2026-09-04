RL with IsaacLab
================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
   :align: center
   :width: 90%

   IsaacLab (image: `IsaacLab <https://developer.nvidia.com/isaac/lab>`__).

`IsaacLab <https://developer.nvidia.com/isaac/lab>`__ is NVIDIA's GPU-accelerated robot
learning simulator. You'll use RLinf to PPO-fine-tune GR00T N1.5 or OpenPI π₀.₅ on a
custom Franka cube-stacking task.

Overview
--------

SFT then PPO-fine-tune a VLA on the IsaacLab Franka stack-cube task.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      GR00T N1.5 · π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Franka stack-cube

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install → download Isaac Sim + an SFT model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Isaac Sim · an SFT checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0``
     - Stack the red block on the blue block, then stack the green block on the red block.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB from a third-person camera and a wrist camera (256×256 by default) plus robot proprioception.
   * - Action
     - 7-dim continuous action: 3D position (x, y, z) + 3D rotation (roll, pitch, yaw) + gripper.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt
     - ``Stack the red block on the blue block, then stack the green block on the red block.``

Installation
------------

.. include:: _setup_common.rst

The Docker image ships venvs only (``/opt/venv``). Bind-mount this checkout to
``/workspace/RLinf`` so an AWS host, a laptop clone, and the container share the
same scripts. Copy the env example once per machine, then use the same
``run_embodiment.sh`` command on every host.

**Docker image**

Build the embodied-isaaclab image once (Blackwell / ``sm_120`` hosts). Skip this
if ``rlinf:embodied-isaaclab-blackwell`` is already present:

.. code:: bash

   bash docker/build_embodied_isaaclab_blackwell.sh

   cp examples/embodiment/scripts/isaaclab_local.env.example \
      examples/embodiment/scripts/isaaclab_local.env
   # Edit ISAAC_SIM_PATH in isaaclab_local.env.

   bash docker/run_embodied_isaaclab_blackwell.sh

What this does:

1. Builds ``rlinf:embodied-isaaclab-blackwell`` with ``install.sh embodied --model openpi --env isaaclab`` (and GR00T).
2. Mounts this checkout at ``/workspace/RLinf`` and Isaac Sim at ``/workspace/isaac_sim``.
3. Opens a login shell with ``/opt/venv/openpi`` on ``PATH``.

Switch to the matching virtual environment inside the image:

.. code:: bash

   # GR00T N1.5
   source switch_env gr00t

   # OpenPI π₀.₅
   # source switch_env openpi

To use a published image instead of a local build, set ``IMAGE_TAG`` in
``isaaclab_local.env`` or on the command line
(``rlinf/rlinf:agentic-rlinf0.3-isaaclab``; mainland China:
``docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-isaaclab``).

**Custom environment**

Install the same model/env combo on the host (laptop or a node without Docker).
Set ``RLINF_NO_DOCKER=1`` in ``isaaclab_local.env`` so ``run_embodiment.sh``
does not re-enter the image:

.. code:: bash

   # Mainland China users can add --use-mirror.

   # GR00T N1.5
   bash requirements/install.sh embodied --model gr00t --env isaaclab
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env isaaclab
   # source .venv/bin/activate

   cp examples/embodiment/scripts/isaaclab_local.env.example \
      examples/embodiment/scripts/isaaclab_local.env
   # Edit ISAAC_SIM_PATH and set RLINF_NO_DOCKER=1.

Machine-local paths
~~~~~~~~~~~~~~~~~~~

``examples/embodiment/scripts/isaaclab_local.env`` is gitignored. Both
``run_embodiment.sh`` and ``docker/run_embodied_isaaclab_blackwell.sh`` source it
when it exists.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Variable
     - Purpose
   * - ``ISAAC_SIM_PATH`` / ``ISAAC_PATH``
     - Isaac Sim 5.1.0 tree. Set one; the loader copies it onto the other name.
   * - ``CRI_OPENPI_CKPT``
     - Converted OpenPI CRI weights. Leave unset to use
       ``checkpoint/pi05_droid_cri_rlinf_49999`` (then ``checkpoints/``).
       Convert JAX ``49999`` with
       ``bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh``.
   * - ``IMAGE_TAG``
     - Docker image for the wrapper and for host re-launch. Leave unset
       to auto-pick ``rlinf:embodied-isaaclab-u24``, then
       ``rlinf:embodied-isaaclab-blackwell``.
   * - ``RLINF_NO_DOCKER``
     - Set ``1`` on a native install so ``run_embodiment.sh`` does not
       re-launch into ``IMAGE_TAG``.

Cube→plate CRI also needs TensorRT 10 (``libnvinfer.so.10``).
``run_embodiment.sh`` calls ``examples/embodiment/scripts/ensure_cri_tensorrt.sh``
and installs into ``.assets/tensorrt`` when missing. Override with
``CRI_EXTRA_LIB_DIRS``.

Download Isaac Sim
~~~~~~~~~~~~~~~~~~

Download Isaac Sim 5.1.0 and initialize its shell environment. The Docker
image does not ship Isaac Sim; install it separately, then set
``ISAAC_SIM_PATH`` in ``isaaclab_local.env`` or place the tree where the
scripts can find it (``./isaac_sim``, this checkout if Sim was extracted
into it, a sibling ``isaac_sim``, or ``/workspace/isaac_sim``).

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   Run ``source ./setup_conda_env.sh`` in every new terminal before launching IsaacLab
   unless you start through ``run_embodiment.sh`` or
   ``docker/run_embodied_isaaclab_blackwell.sh`` (both source it for you).

Download the Model
------------------

Download the checkpoint for the model you plan to fine-tune.

**GR00T N1.5**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Stack-cube

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir RLinf-Gr00t-SFT-Stack-cube

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/YifWRobotics/RLinf-pi05-SFT-Stack-cube

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download YifWRobotics/RLinf-pi05-SFT-Stack-cube --local-dir RLinf-pi05-SFT-Stack-cube

.. include:: _model_path.rst

The SFT checkpoints come from human demonstrations collected on the IsaacLab stack-cube
task. The dataset is available on |huggingface|
`IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__.

Run It
------

Pick one config and launch training:

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - Model
     - Config
     - Command suffix
   * - GR00T N1.5
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``
     - ``isaaclab_franka_stack_cube_ppo_gr00t``
   * - OpenPI π₀.₅
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml``
     - ``isaaclab_franka_stack_cube_ppo_openpi_pi05``
   * - OpenPI π₀.₅ CRI (DROID joint-pos, cube→plate)
     - ``examples/embodiment/config/isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri.yaml``
     - ``isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri``

.. code:: bash

   # GR00T N1.5
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

   # OpenPI π₀.₅
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05

   # OpenPI π₀.₅ from pi05_droid_cri_finetune (DROID 8-D, cube→plate)
   # 1) Convert JAX/LoRA weights if you only have the Orbax ``49999`` tree:
   #    bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh
   #    (writes checkpoint/pi05_droid_cri_rlinf_49999)
   # 2) EnvWorker registers Isaac-PickPlace-Cube-Plate-Droid-AbsJointPos-v0 from
   #    rlinf/envs/isaaclab/tasks/pick_place_cube_plate (Arena DROID specs, single layout).
   #    Each episode samples exterior_1 vs exterior_2 50:50 into the policy base
   #    image, matching openpi DROID RLDS training.
   #    Online CRI(q, qd) is tokenized as a discrete VLM span (droid-cri / 49999).
   # The dedicated launcher enters the local embodied-isaaclab image
   # (rlinf:embodied-isaaclab-u24, then rlinf:embodied-isaaclab-blackwell)
   # and exports CRI_OPENPI_CKPT from checkpoint/pi05_droid_cri_rlinf_49999
   # (or the older checkpoints/ path). TensorRT 10 (libnvinfer.so.10) is
   # installed into .assets/tensorrt when missing; override with CRI_EXTRA_LIB_DIRS.
   bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh
   # equivalent:
   # bash examples/embodiment/run_embodiment.sh isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri

What this does:

1. Starts the embodied training entrypoint with the selected Hydra config.
2. Creates Ray workers for the actor, rollout, and IsaacLab env components.
3. Runs PPO rollouts, computes sparse task rewards, and updates the VLA policy.

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with config fallback and the same suffixes:
``isaaclab_franka_stack_cube_ppo_gr00t`` and
``isaaclab_franka_stack_cube_ppo_openpi_pi05``.

.. note::

   For GR00T, the default config separates env, rollout, and actor placement. For OpenPI,
   the default config collocates ``actor,env,rollout: all``. Tune
   ``cluster.component_placement``, ``rollout.pipeline_stage_num``, and
   ``actor.enable_offload`` for your GPU memory budget.

.. note::

   To add a custom IsaacLab task, implement it under
   ``rlinf/envs/isaaclab/tasks/``, register it in ``rlinf/envs/isaaclab/__init__.py``,
   then point ``init_params.id`` in an env config such as
   ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` at the new task id.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when you want rollout videos:

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

Enable W&B or SwanLab by adding logger backends:

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Model Stage
     - Success Rate
   * - GR00T N1.5 base model (no SFT)
     - 0.000
   * - GR00T N1.5 SFT model
     - 0.654
   * - GR00T N1.5 RL-tuned model (SFT + RL)
     - 0.897
   * - OpenPI π₀.₅ SFT model
     - 0.859
   * - OpenPI π₀.₅ RL-tuned model (SFT + RL)
     - 0.953

Acknowledgements
----------------

Credit to `Minghui Xu <https://github.com/smallcracker>`__ and
`Nan Yang <https://github.com/AquaSage18>`__ for the GR00T N1.5 example, and
`Yifan Wu <https://github.com/YifWRobotics>`__ for the OpenPI π₀.₅ example.
