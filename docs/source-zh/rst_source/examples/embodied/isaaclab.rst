基于 IsaacLab 的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
   :align: center
   :width: 90%

   IsaacLab（图片来源：`IsaacLab <https://developer.nvidia.com/isaac/lab>`__）。

`IsaacLab <https://developer.nvidia.com/isaac/lab>`__ 是 NVIDIA 的 GPU 加速机器人学习仿真器。
你将使用 RLinf 在自定义 Franka 方块堆叠任务上，通过 PPO 微调 GR00T N1.5 或 OpenPI π₀.₅。

概览
----------------------------------------

先使用 SFT 检查点，再通过 PPO 在 IsaacLab Franka stack-cube 任务上微调 VLA。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      GR00T N1.5 · π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      Franka stack-cube

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPUs

| **你将完成：** 安装 → 下载 Isaac Sim + SFT 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Isaac Sim · SFT 检查点（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0``
     - 将红色方块堆到蓝色方块上，再将绿色方块堆到红色方块上。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 第三人称相机和腕部相机 RGB（默认 256×256），以及机器人本体状态。
   * - 动作
     - 7 维连续动作：3D 位置（x, y, z）+ 3D 旋转（roll, pitch, yaw）+ 夹爪。
   * - 奖励
     - 稀疏 0/1 成功奖励。
   * - 提示词
     - ``Stack the red block on the blue block, then stack the green block on the red block.``

安装
----------------------------------------

.. include:: _setup_common.rst

Docker 镜像只提供 venv（``/opt/venv``）。把当前 checkout 绑定到
``/workspace/RLinf``，这样 AWS 主机、笔记本克隆和容器会共用同一套脚本。
每台机器复制一次 env 示例后，所有主机都用同一个 ``run_embodiment.sh`` 命令。

**Docker 镜像**

先构建 embodied-isaaclab 镜像一次（Blackwell / ``sm_120`` 主机）。
若已有 ``rlinf:embodied-isaaclab-blackwell`` 则跳过：

.. code:: bash

   bash docker/build_embodied_isaaclab_blackwell.sh

   cp examples/embodiment/scripts/isaaclab_local.env.example \
      examples/embodiment/scripts/isaaclab_local.env
   # 在 isaaclab_local.env 中修改 ISAAC_SIM_PATH。

   bash docker/run_embodied_isaaclab_blackwell.sh

这条命令会：

1. 用 ``install.sh embodied --model openpi --env isaaclab``（以及 GR00T）构建 ``rlinf:embodied-isaaclab-blackwell``。
2. 把当前 checkout 挂到 ``/workspace/RLinf``，把 Isaac Sim 挂到 ``/workspace/isaac_sim``。
3. 打开登录 shell，并将 ``/opt/venv/openpi`` 加入 ``PATH``。

在镜像中切换到对应的虚拟环境：

.. code:: bash

   # GR00T N1.5
   source switch_env gr00t

   # OpenPI π₀.₅
   # source switch_env openpi

若不用本地构建、改用已发布镜像，在 ``isaaclab_local.env`` 或命令行设置
``IMAGE_TAG``（``rlinf/rlinf:agentic-rlinf0.3-isaaclab``；国内：
``docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-isaaclab``）。

**自定义环境**

在主机上安装同一组 model/env（笔记本或无 Docker 的节点）。
在 ``isaaclab_local.env`` 里设置 ``RLINF_NO_DOCKER=1``，避免
``run_embodiment.sh`` 再次进入镜像：

.. code:: bash

   # 国内用户可添加 --use-mirror。

   # GR00T N1.5
   bash requirements/install.sh embodied --model gr00t --env isaaclab
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env isaaclab
   # source .venv/bin/activate

   cp examples/embodiment/scripts/isaaclab_local.env.example \
      examples/embodiment/scripts/isaaclab_local.env
   # 修改 ISAAC_SIM_PATH，并设置 RLINF_NO_DOCKER=1。

机器本地路径
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/embodiment/scripts/isaaclab_local.env`` 已被 gitignore。
``run_embodiment.sh`` 与 ``docker/run_embodied_isaaclab_blackwell.sh``
在文件存在时会 source 它。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 变量
     - 用途
   * - ``ISAAC_SIM_PATH`` / ``ISAAC_PATH``
     - Isaac Sim 5.1.0 目录。只需设置其中一个，加载脚本会同步到另一个名字。
   * - ``CRI_OPENPI_CKPT``
     - 转换后的 OpenPI CRI 权重。留空则使用
       ``checkpoint/pi05_droid_cri_rlinf_49999``（然后是 ``checkpoints/``）。
       JAX ``49999`` 请用
       ``bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh`` 转换。
   * - ``IMAGE_TAG``
     - Docker 包装脚本以及主机再启动时使用的镜像。留空则依次选择
       ``rlinf:embodied-isaaclab-u24``、
       ``rlinf:embodied-isaaclab-blackwell``。
   * - ``RLINF_NO_DOCKER``
     - 在本地安装上设为 ``1``，避免 ``run_embodiment.sh`` 再进入 ``IMAGE_TAG``。

方块→盘子 CRI 还需要 TensorRT 10（``libnvinfer.so.10``）。
``run_embodiment.sh`` 会调用 ``examples/embodiment/scripts/ensure_cri_tensorrt.sh``，
缺失时安装到 ``.assets/tensorrt``。也可用 ``CRI_EXTRA_LIB_DIRS`` 覆盖。

下载 Isaac Sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 Isaac Sim 5.1.0 并初始化其 shell 环境。Docker 镜像不包含 Isaac Sim，
需单独安装，然后在 ``isaaclab_local.env`` 中设置 ``ISAAC_SIM_PATH``，
或把目录放在脚本能找到的位置（``./isaac_sim``、Sim 被解压到当前 checkout、
同级 ``isaac_sim``、或 ``/workspace/isaac_sim``）。

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   每次在新终端中启动 IsaacLab 前，都需要运行 ``source ./setup_conda_env.sh``。
   若通过 ``run_embodiment.sh`` 或 ``docker/run_embodied_isaaclab_blackwell.sh``
   启动，脚本会替你 source。

下载模型
----------------------------------------

下载你要微调的模型检查点。

**GR00T N1.5**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Stack-cube

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir RLinf-Gr00t-SFT-Stack-cube

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/YifWRobotics/RLinf-pi05-SFT-Stack-cube

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download YifWRobotics/RLinf-pi05-SFT-Stack-cube --local-dir RLinf-pi05-SFT-Stack-cube

.. include:: _model_path.rst

这些 SFT 检查点来自 IsaacLab stack-cube 任务的人类演示数据。
数据集已发布在 |huggingface|
`IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__。

运行
----------------------------------------

选择一个配置并启动训练：

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - 模型
     - 配置
     - 命令后缀
   * - GR00T N1.5
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``
     - ``isaaclab_franka_stack_cube_ppo_gr00t``
   * - OpenPI π₀.₅
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml``
     - ``isaaclab_franka_stack_cube_ppo_openpi_pi05``
   * - OpenPI π₀.₅ CRI（DROID joint-pos，立方体→盘子）
     - ``examples/embodiment/config/isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri.yaml``
     - ``isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri``

.. code:: bash

   # GR00T N1.5
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

   # OpenPI π₀.₅
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05

   # OpenPI π₀.₅（以 pi05_droid_cri_finetune 为起点，DROID 8-D，方块→盘子）
   # 1) 若只有 Orbax ``49999`` 树，先转成 RLinf 可加载的 safetensors：
   #    bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh
   #    （写入 checkpoint/pi05_droid_cri_rlinf_49999）
   # 2) EnvWorker 会从 rlinf/envs/isaaclab/tasks/pick_place_cube_plate 注册
   #    Isaac-PickPlace-Cube-Plate-Droid-AbsJointPos-v0（Arena DROID 规格、单布局）。
   #    每个 episode 以 50:50 将 exterior_1 / exterior_2 采样到策略 base 图像，
   #    与 openpi DROID RLDS 训练一致。
   #    控制频率为 15 Hz（OpenPI DROID / action_horizon=15），不是 IsaacLab 默认 50 Hz。
   #    在线 CRI(q, qd) 会编成离散 VLM span（droid-cri / 49999）。
   # 专用启动脚本会进入本机 embodied-isaaclab 镜像
   # （先 rlinf:embodied-isaaclab-u24，再 rlinf:embodied-isaaclab-blackwell），
   # 并从 checkpoint/pi05_droid_cri_rlinf_49999（或旧的 checkpoints/）
   # 自动导出 CRI_OPENPI_CKPT。
   # 缺少 TensorRT 10（libnvinfer.so.10）时会装到 .assets/tensorrt；
   # 也可用 CRI_EXTRA_LIB_DIRS 指定。
   bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh
   # 等价写法：
   # bash examples/embodiment/run_embodiment.sh isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri

这条命令会：

1. 使用选定的 Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 IsaacLab env 组件创建 Ray worker。
3. 运行 PPO rollout，计算稀疏任务奖励，并更新 VLA 策略。

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>`，
通过配置回退机制复用相同后缀：``isaaclab_franka_stack_cube_ppo_gr00t`` 和
``isaaclab_franka_stack_cube_ppo_openpi_pi05``。

.. note::

   GR00T 默认配置会分离 env、rollout 和 actor placement。OpenPI 默认配置使用
   ``actor,env,rollout: all`` 共置。请根据 GPU 显存预算调整
   ``cluster.component_placement``、``rollout.pipeline_stage_num`` 和
   ``actor.enable_offload``。

.. note::

   如需添加自定义 IsaacLab 任务，请在 ``rlinf/envs/isaaclab/tasks/`` 下实现任务，
   在 ``rlinf/envs/isaaclab/__init__.py`` 中注册任务，然后在
   ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` 等环境配置中，将
   ``init_params.id`` 指向新的 task id。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。完整指标说明见
:doc:`训练指标 <../../reference/metrics>`。

如需保存 rollout 视频，请在环境配置中启用 video：

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

如需启用 W&B 或 SwanLab，请添加 logger backend：

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - 模型阶段
     - 成功率
   * - GR00T N1.5 基础模型（无 SFT）
     - 0.000
   * - GR00T N1.5 SFT 模型
     - 0.654
   * - GR00T N1.5 RL 微调模型（SFT + RL）
     - 0.897
   * - OpenPI π₀.₅ SFT 模型
     - 0.859
   * - OpenPI π₀.₅ RL 微调模型（SFT + RL）
     - 0.953

致谢
----------------------------------------

感谢 `许明辉 <https://github.com/smallcracker>`__ 和
`杨楠 <https://github.com/AquaSage18>`__ 对 GR00T N1.5 示例的贡献与支持，也感谢
`Yifan Wu <https://github.com/YifWRobotics>`__ 对 OpenPI π₀.₅ 示例的贡献与支持。
