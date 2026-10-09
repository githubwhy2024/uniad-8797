# UniAD 8797 development

This standalone repository starts from the final Q4 code state. The user authorized its public initialization and resource upload on 2026-10-09. Read README.md, docs/Q5_HANDOFF.md and docs/Q5_PLAN.md before development.

- Modify this repository only. Other sibling projects and their shared resources remain read-only.
- Preserve original UniAD and dynamic sources unless the user explicitly requests a change. Keep fixed changes in their responsible existing files. Both variants retain the matching 13 Python files.
- Bind the actual code, model, initial state, input, ABI and backend hashes before expensive work. Keep checker, conversion, compilation, load, real recurrence, task metrics and board acceptance separate. Failures remain failed_acceptance; process exit zero alone is insufficient.
- Production applications return final post-processed planning plus frame identity/validity. State feedback, count/mask/overflow and fail-closed transaction checks remain until demonstrated unnecessary.
- Q4 host evidence is historical reference. The three-part/new-Host candidate has no full task acceptance. CPU timing, 18 GiB budget, x86 layout and partition count do not specify 8797 behavior.
- Confirm the actual SDK, OS/BSP/driver, toolchain, backend, precision and resource limits before selecting a target configuration. Quantization/FP16 require an explicit decision informed by target behavior.
- Write new outputs under onnx/runs/<purpose>-<random>. Keep licenses and source provenance. Do not upload datasets, credentials, SDK installations or host backend libraries as part of routine commits.
- Update docs/Q5_HANDOFF.md and commit at meaningful milestones. For long work record command/cwd/environment/identities/PID/status/log/result and next check. No new NN inference or board development is part of this initialization.
