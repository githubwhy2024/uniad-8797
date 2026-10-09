# 环境、资源和数据

## 已有主机参考

历史参考使用 Python 3.9.25、NumPy 1.22.4、ONNX 1.13.1、ONNX Runtime 1.19.2、CasADi 3.6.7；完整 PT/导出/任务工具还需要匹配的 PyTorch、CUDA、MMCV/MMDet/MMDet3D、nuScenes devkit 与项目 requirements.txt。请结合实际 SDK 和机器建立环境，不能把 x86 环境直接当成 ARM 环境。运行时最小版本在 onnx/qnn/requirements-runtime.txt。获取和 SHA 校验资源只需要标准库。

## SDK 配置

当前公开版主机开发参考为 QAIRT **2.42.0.251225**，已验证工具 build 为 `v2.42.0.251225135753_193295`。旧 CPU 指标来自 2.31，保留为历史来源。公开 2.42 的主机验证不代表 QNX/8797 板端接受；该 ZIP 没有 QNX Add-On。

每人独立安装 SDK，安装目录、Conda 目录与环境名可以不同。仓库不携带 SDK，也不提交任何人的绝对安装路径。以下步骤用于 Linux/WSL Bash；其他主机系统应先确认对应工具与 ABI。

首次创建转换环境（已有合适环境可跳过）：

```bash
conda env create -f configs/environment.qualcomm.yml
conda activate qualcomm
```

`configs/requirements-qualcomm.txt` 是已验证的 Python 3.10 主机依赖选择。protobuf 3.20.3 满足 ONNX 的最低要求；setuptools 80.9.0 保留 SDK 检查所用的 pkg_resources。它们与 SDK tested-version 清单有部分差别。环境文件的默认名称为 qualcomm，同事可用 `conda env create -n 自己的名称 -f configs/environment.qualcomm.yml` 覆盖。

将**自己机器上的 SDK 目录**绑定到当前 Conda 环境，只需执行一次：

```bash
python tools/install_sdk_hooks.py --sdk /your/sdk/2.42.0.251225
conda deactivate
conda activate qualcomm
qnn-net-run --version
```

之后只需激活这个 Conda 环境。钩子自动设置 `QAIRT_SDK`、`QAIRT_SDK_ROOT`、`QNN_SDK_ROOT`、`QNN_CONVERTER_ENV` 和必要 Python/库路径；工具链接及绝对路径只保存在本机 Conda 目录。正常退出或切换环境时恢复进入前的 SDK/Python/库变量，包括原来未设置或为空的状态。重复安装会更新本工具拥有的链接，拒绝覆盖其他工具或钩子。不要在保持此环境已激活时重新绑定另一份 SDK，应先退出再安装。

QNN 构建工具优先读取 `QAIRT_SDK`，也支持 SDK 官方变量 `QAIRT_SDK_ROOT` / `QNN_SDK_ROOT`；转换环境默认使用当前 Python 前缀。C++ bridge 默认在 PATH 查找 clang++，可通过 `CXX` 或 `build_bridge.py --cxx` 指定。

完整 PT、导出、Host 和任务评测仍需单独的 UniAD 环境。跨环境编排时显式设置 **自己机器上的** 模型 Python：

```bash
export UNIAD_PYTHON=/your/model/environment/bin/python
```

该变量不由转换环境猜测，也不会把转换环境当作已具备 PyTorch/MMCV 的模型环境。个人覆盖项可放在 Git 忽略的 `.local/env.sh`，由本人显式 source。

不使用 Conda 钩子时，仍可手动选择：

```bash
export QAIRT_SDK=/your/matching/qairt
export QNN_CONVERTER_ENV=/your/converter/environment
source onnx/qnn/env.sh
```

`--backend` 会覆盖 CPU 控制的默认库；缺省库来自当前 SDK，不回退到旧 2.31。这些控制和构建脚本仍明确面向 x86 CPU，目标 ABI、bridge、context、内存注册和后端需要 Q5 实现与验收。

规划资源读取 `onnx/qnn/planning_bundle_spec.template.json`，其中相对路径按当前仓库根目录解析，不依赖启动目录。个人资源配置可通过 `UNIAD_BUNDLE_SPEC=/your/local/spec.json` 覆盖。模板只是资源/导出说明，不授予 QNN 或板端接受。实际目标配置另行复制 `configs/target.example.json` 到 `.local/` 填写；板端版本、BSP、driver 和后端须绑定实际设备，保持与主机 SDK 配置分开。

## 数据

从 https://www.nuscenes.org/nuscenes 按官方流程获取 nuScenes mini，并按照原始 UniAD 数据准备方式生成 info 文件，放在本仓库 `data/nuscenes/`。mini_val 为 2 场景/81 帧，完整 mini 为 10 场景/404 帧。数据集和原始图像不随本公开仓库分发；数据准备与使用遵循对应数据许可。

Release 包含 `data/others/motion_anchor_infos_mode6.pkl`。PT 构建显式使用 `ckpts/uniad_base_e2e.pth`；配置里的 training `load_from` 不是本基线推理权重入口。

## 入口边界

fixed/dynamic 源码保持原来源身份，部分历史 promotion/export 子命令需要原有完整证据树，它们不作为本仓库的初始化入口。开发以 Release 中已验逻辑 ONNX、资源清单和已迁入的指标参考起步；新导出与输入清单必须为当前代码重新绑定来源。旧报告中的路径/PID仅是历史数据，不表示本机仍在运行。

下载后先执行 `tools/check_setup.py`，不要直接照搬历史 CPU 命令开始长推理。新输出放 `onnx/runs/`，不覆盖基线资源。Host/固定状态及通用数学接口保持冻结规则。

当前已核对的 PT 参考版本：torch 2.0.1+cu118、mmcv-full 1.6.1、mmdet 2.26.0、mmdet3d 1.0.0rc6、nuScenes devkit 1.2.0。它们是当前主机版本记录，不是 8797 安装指令。
