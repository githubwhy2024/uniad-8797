# 主机环境与便携配置验证

日期：2026-10-09。SDK 实测 build：`v2.42.0.251225135753_193295`。

| 验证 | 结果 |
| --- | --- |
| 本机 Conda 自动激活与重复激活 | 2.42；Python 来自所激活的转换环境，SDK 路径无重复 |
| 退出与切换到模型环境 | 恢复进入前变量，区分未设置与空值；不残留 2.42 工具 PATH |
| 手动旧 SDK 与自动新 SDK 切换 | 2.31 → 2.42 → 2.31，当前工具/库路径匹配所选 SDK |
| 第二台电脑配置模拟 | 不同 SDK/Conda 目录，包含空格；安装、重复安装、变量恢复通过 |
| 工具冲突保护 | 拒绝覆盖非本安装器管理的命令 |
| Q5 实际诊断转换入口 | Conv 1×1 + ReLU 小模型，FP32 转换及边界元数据审计通过 |
| 2.42 元数据兼容 | 缺失 axis_format 记为未报告；记录 permute_order_to_src，保留原 dtype/尺寸/静态边界检查 |
| 元数据拒绝控制 | 错误 dtype、错误非 singleton 尺寸、动态边界均拒绝 |
| 实际 x86 模型库编译 | 通过 |
| 实际 bridge 编译 | 使用 PATH 中的 clang++ 与 2.42 头文件，通过 |
| 工具生命周期 | 成功/失败/SIGTERM/任务拒绝等 32 项检查通过，failed_acceptance 保留 |

首次 Q5 转换尝试因审计读取旧 axis_format 字段失败，证据保留；修复后在新目录重新验证。SDK converter 本身在首次尝试也成功，失败没有改写成通过。

最终本机证据目录为 `onnx/runs/q5-sdk-config-2c6339f4/`。审计、编译结果哈希：

- converter audit：`d85d8cbfe1c9bb06680a15ebfd872cf075c35cf0d7a065ab9ab40a111a1463ef`
- model library：`e9146adfc563d1842c099e997a8e2301f3d20d5ed30e7d954697a305ef2dc475`
- bridge library：`381f83d1de291bef4663d5c5242aa0644554847d065f1b31f9b3c11e1bcb0c3a`

该目录由各人的实际运行生成，不随 Git 分发。钩子与配置检查属于主机准备；此次没有执行小模型/UniAD 神经网络，没有重做完整任务指标，没有 QNX/HTP 板端执行或 10 Hz 接受。既有逻辑模型、权重、初始状态、fixed/dynamic/原始 UniAD 源码及历史指标哈希保持。
