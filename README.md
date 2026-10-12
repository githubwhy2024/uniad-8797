# UniAD on Qualcomm 8797

协作开发仓库，以 Q4 收尾源码 `45df36449e38fa22eeee13c2337e8fd281edd784` 为起点，开展 8797 的目标适配与验证。默认开发分支 `qualcomm/q5-8797`，`main` 保存共同基线。

## 获取代码与必要资源

```bash
git clone https://github.com/githubwhy2024/uniad-8797.git
cd uniad-8797
git switch qualcomm/q5-8797
python3 tools/fetch_resources.py
python3 tools/check_setup.py --baseline-sources
```

代码和清单由 Git 管理；两个 ONNX、初始状态、模型权重、motion anchor 和证明文件由同仓库 Release `q5-baseline-v1` 提供。下载脚本逐个校验大小与 SHA256。资源清单在 [manifest.json](onnx/resources/manifest.json)。无需 Git LFS。

已有离线文件时运行 `python3 tools/fetch_resources.py --from-dir /path/to/release-assets`；仅核验时加 `--verify-only`。安装对应 ONNX/NumPy 环境后运行 `python tools/check_setup.py --baseline-sources --onnx`，检查两个模型和初始状态；这一步不执行神经网络。

## 本机 SDK 默认选择

Linux/WSL 主机开发使用公开 QAIRT 2.42。每人按 [环境说明](docs/SETUP.md) 为自己的 Conda 环境安装一次 `tools/install_sdk_hooks.py` 钩子，之后只需 `conda activate qualcomm`（或自己的环境名）。SDK、转换环境、模型 Python 和编译器路径均由本机配置；仓库不固定开发者安装位置。SDK 和 QNX Add-On 需要各自取得，不随资源 Release 分发。

## 开发入口

- [新人员说明](docs/HOST_PREPARATION.md)：已有机制、CPU 兼容处理与目标替换原则。
- [Q5 计划](docs/Q5_PLAN.md)、[交接](docs/Q5_HANDOFF.md)、[环境与数据](docs/SETUP.md)。
- `onnx/fixed/` 和 `onnx/dynamic/`：各 13 个 Python 文件，原始 UniAD 在 `projects/`。
- `onnx/qnn/`：通用图/生命周期合同与已有 CPU 参考实现。`backend_contract.py` 说明适配器的保留、跳过和替换条件。
- `onnx/validation/`：必要的结构、接口、状态、包和任务工具。
- `onnx/reference/`：冻结指标、适配函数和来源清单。历史绝对路径仅作为来源标识，通过映射定位到本仓库参考文件。

规划逻辑模型为 23 输入/25 输出；六任务参考为 23 输入/47 输出。生产结果需要 Host 的最终后处理计划，同时保留状态与有效性合同。已有 x86 构建工具用于主机参考，目标工具链实现仍待确认。

## 当前验收范围

已有完整六任务与规划主机结果作为冻结参考。三段/新 Host 完整任务 `not_evaluated`，8797 运行及 10 Hz `not_evaluated`。本初始化不运行新 NN，不生成目标 context，也不将 CPU 产物作为板端产物。

## 两人协作

从 `qualcomm/q5-8797` 各建自己的功能分支，通过 PR 汇合；资源变更使用新 Release 标签、清单和哈希，保留旧版本。共享基线由双方约定后更新到 `main`。公开仓库可直接下载；直接推送权限需另行邀请协作者，也可使用 fork/PR。

本仓库为独立 Git 仓库，origin 只关联本仓库。Apache-2.0 及第三方版权声明见 LICENSE 和 onnx/QUALCOMM_THIRD_PARTY_LICENSE.txt；资源来源见 provenance.json。
