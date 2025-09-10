# FunASR-bmodel for monitor

## 快速开始

```sh
uv sync
source .venv/bin/activate

bash download.sh

cd bmodel
python monitor.py   --rtsp rtsp://172.24.64.225:8554/stream1   --system-audio ../system_ref.wav  --system-phrases ../system_phrases.txt   --chunk-seconds 10
```

## 说明

1. 本python项目使用uv管理，请先安装uv。参考安装方式：

```bash
# 推荐方式
curl -LsSf https://astral.sh/uv/install.sh | sh

# pipx 安装
sudo apt install pipx
pipx install uv
pipx ensurepath
```

2. 虚拟环境创建/同步

   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. 本项目需要下载对应的模型，可以使用脚本 `bash download.sh` 快速下载。
4. 关键文件

   - 模型推理脚本 `bmodel/offline.py` ，直接运行即可。
   - 对话监测脚本 `bmodel/monitor.py` ，运行方式如下

     ```bash
     cd bmodel
     python monitor.py   --rtsp rtsp://172.24.64.225:8554/stream1   --system-audio ../system_ref.wav  --system-phrases ../system_phrases.txt   --chunk-seconds 10
     ```

    相关参数说明：

    ```python
       parser = argparse.ArgumentParser(description="监控场景下非系统人声检测与上报")
       parser.add_argument("--rtsp", required=True, help="RTSP 地址，例如 rtsp://user:pass@ip:554/Streaming/Channels/101")
       parser.add_argument(
           "--system-audio",
           default="../system_ref.wav",
           help="系统参考音频，作为前置拼接，系统说话人 spk=0",
       )
       parser.add_argument("--out-dir", default="/data/alarms", help="告警本地保存目录 (已由 nginx 映射到 /static)")
       parser.add_argument("--chunk-seconds", type=int, default=10, help="检测窗口秒数，默认 10s")
       parser.add_argument(
           "--system-phrases",
           default="../system_phrases.txt",
           help="系统提示词文本库（每行一条），用于过滤系统播报文本。如果系统声音错误上报，请将对应的识别文本加入到该文件。",
       )
   ```

   还有一些参数没有需要直接修改代码：
   - ip修改：搜索并修改`get_local_ip_fallback()`
   - POST信息修改：搜索并修改`def post_alert(safety_url: str, when: str, timeout: int = 2)`
