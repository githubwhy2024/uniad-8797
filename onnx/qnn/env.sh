#!/usr/bin/env bash
# Optional manual setup; Conda hooks can provide the same variables automatically.
_uniad_qnn_env() {
    local sdk="${QAIRT_SDK:-${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-}}}"
    local prefix="${QNN_CONVERTER_ENV:-${CONDA_PREFIX:-}}" key entry value
    local -a entries
    if [[ ! -f "$sdk/bin/x86_64-linux-clang/qnn-onnx-converter" || ! -x "$prefix/bin/python" ]]; then
        printf '%s\n' 'Activate your configured Conda environment, or set QAIRT_SDK and QNN_CONVERTER_ENV.' >&2
        return 1
    fi
    export QAIRT_SDK="$sdk" QAIRT_SDK_ROOT="$sdk" QNN_SDK_ROOT="$sdk" QNN_CONVERTER_ENV="$prefix"
    for key in PATH PYTHONPATH LD_LIBRARY_PATH; do
        value=''
        IFS=: read -r -a entries <<< "${!key-}"
        for entry in "${entries[@]}"; do
            case "$entry" in
                "$prefix/bin"|"$sdk/bin/x86_64-linux-clang"|"$sdk/lib/python"|"$sdk/lib/python/"|"$sdk/lib/x86_64-linux-clang"|'') ;;
                *) value="${value:+$value:}$entry" ;;
            esac
        done
        case "$key" in
            PATH)
                if [[ "$(readlink -f "$prefix/bin/qnn-onnx-converter" 2>/dev/null)" == "$sdk/bin/x86_64-linux-clang/qnn-onnx-converter" ]]; then
                    value="$prefix/bin${value:+:$value}"
                else value="$prefix/bin:$sdk/bin/x86_64-linux-clang${value:+:$value}"; fi ;;
            PYTHONPATH) value="$sdk/lib/python${value:+:$value}" ;;
            LD_LIBRARY_PATH) value="$sdk/lib/x86_64-linux-clang${value:+:$value}" ;;
        esac
        printf -v "$key" '%s' "$value"
        export "$key"
    done
}
_uniad_qnn_env
_uniad_qnn_status=$?
unset -f _uniad_qnn_env
return "$_uniad_qnn_status" 2>/dev/null || exit "$_uniad_qnn_status"
