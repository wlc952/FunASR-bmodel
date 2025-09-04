# FunASR-bmodel for monitor

## 使用说明

```sh
uv sync
source .venv/bin/activate

bash download.sh

cd bmodel
python monitor.py   --rtsp rtsp://172.24.64.225:8554/stream1   --system-audio ../system_ref.wav  --system-phrases ../system_phrases.txt   --chunk-seconds 10
```
