# Q5 初始交接

2026-10-09。本仓库从 Q4 收尾提交 `45df36449e38fa22eeee13c2337e8fd281edd784` 创建，在并列 worktree 中清理后转为独立 Git 仓库。源仓库与其他 worktree 保持原远端和源码状态。

## 起点

- 本地路径 `uniad_onnx_qualcomm_q5`；独立 `.git/`；默认 `main`，工作分支 `qualcomm/q5-8797`。
- 原始 UniAD 与 fixed/dynamic 各 13 个 Python 文件保留。QNN CPU 兼容实现保留为参考，已移除 Q4 分支守卫、2.31 运行默认值和本机环境默认路径；通过本机 Conda 钩子或明确的环境变量绑定工具。
- 必要的冻结输入/任务适配函数按 AST 原样迁入，完整参考指标及其原哈希保留。历史报告只作为不可变来源，不指示本机继续旧运行。
- 模型、初始状态、权重、motion anchor、证明和后处理资源通过清单与 Release 绑定，均可恢复为普通文件；无需旧项目软链接。
- 历史 run、CPU 编译库、临时包、清理脚本、旧规划/交接流水已从此副本移除；原 Q4 文件完整保留。

## 验收边界

原 47 六任务及旧 214 段规划任务结果是冻结主机参考。三段/新 Host 完整任务仍为 `not_evaluated`，目标 8797/HTP/10 Hz 也为 `not_evaluated`。本初始化只进行来源、下载/校验、接口、结构和初始状态检查，没有运行新神经网络或目标开发。

## 接手顺序

先执行 README 中的资源获取与检查，再读 SETUP 和 Q5_PLAN。先确认实际目标 SDK/OS/BSP/driver/工具链/后端/精度，完善 configs/target.example.json。旧 x86 桥接与构建工具不直接生成已接受的 8797 产物。

`onnx/qnn/planning_bundle_spec.template.json` 中的路径均为仓库相对路径，由资源读取器按当前仓库根目录解析，可通过个人配置覆盖；它是导出说明模板。生产 QNN 包需重新生成 pin，并绑定实际目标资源，不能复用 ORT 包证书。

## 本次起点修复

主机开发参考改为公开 QAIRT 2.42；诊断 converter、CPU 控制默认 backend、工具身份与规划包 SDK 版本均从实际安装取得，C++ 编译器由 PATH/CXX/参数选择。规划资源模板按仓库根解析，可用 UNIAD_BUNDLE_SPEC 覆盖。环境安装脚本只在各人的 Conda 目录生成本机配置，日常无需再次 source 项目环境脚本。公开 SDK 缺少 QNX Add-On，主机配置不构成目标配置或板端接受。

按用户要求，main、qualcomm/q5-8797 和 q5-baseline-v1 标签以修复后的同一根提交重新初始化。模型/权重/证明与 Release 附件哈希不变，历史 Q4 结果仍标明 2.31 来源。没有新 NN 或板端执行；当前验证记录见 docs/ENVIRONMENT_VALIDATION.md。

源文件快照与模型哈希见 provenance.json、onnx/reference/source_snapshot.json 和 onnx/resources/manifest.json。修改源码后基线 hash 检查会失败是预期行为，应绑定新的候选身份与证据。
