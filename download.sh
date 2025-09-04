#!/usr/bin/env bash
set -euo pipefail
dirs=(
  bmodel/punc_ct-transformer_zh-cn-common-vocab272727/scripts
  bmodel/speech_campplus_sv_zh-cn_16k-common/scripts
  bmodel/speech_fsmn_vad_zh-cn-16k-common/scripts
  bmodel/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404/scripts
)
for d in "${dirs[@]}"; do
  echo "==> $d"
  test -f "$d/download.sh" || { echo "missing $d/download.sh"; exit 1; }
  chmod +x "$d/download.sh"
  (cd "$d" && ./download.sh)
done
echo "All done."
