#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
监控场景语音检测：
- 使用 ffmpeg 将 RTSP 流按 10s 切片为 mp4（含音频）。
- 对每个切片提取 16k 单声道 wav，并在前面拼接系统参考音频（系统说话人 spk=0）。
- 调用 FunASR AutoModel 进行识别与说话人区分，若存在非0说话人，且文本不在系统提示词库内，则判定为告警。
- 命中后将对应 10s mp4 保存到本地 /data/alarms 下，并按协议 POST 上报。

运行示例：
  python bmodel/monitor.py \
    --rtsp rtsp://user:pass@ip:554/Streaming/Channels/101 \
    --system-audio /data/FunASR-bmodel/system_ref.wav \
    --chunk-seconds 10 \
    --out-dir /data/alarm

注意：
- 本机需已安装 ffmpeg 且可访问 RTSP。
- nginx 已将 /data/alarm 映射至 http://<ip>/static/。
"""

import argparse
import datetime as dt
import requests
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import queue
import threading
import time
from typing import List, Optional, Tuple

# 使得可以从项目根目录导入 funasr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from funasr import AutoModel  # type: ignore


# ----------------------------- 工具函数 -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_local_ip_fallback() -> str:
    ip = "127.0.0.1"
    return ip


def normalize_text(text: str) -> str:
    """简单文本归一化：去空白、常见标点，转小写。"""
    text = text.strip().lower()
    # 去除中英文标点与空白
    text = re.sub(r"[\s\u3000\t\r\n]+", "", text)
    text = re.sub(r"[，。！？、；：‘’“”'\".,!?;:()\[\]{}<>-]", "", text)
    return text


def longest_common_substring_len(a: str, b: str) -> int:
    """计算最长公共子串长度（按字符计数，中文自然计为1个字符）。"""
    s1 = normalize_text(a)
    s2 = normalize_text(b)
    if not s1 or not s2:
        return 0
    n, m = len(s1), len(s2)
    # DP 优化为一维，记录上一行
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def load_system_phrases(path: Optional[str]) -> List[str]:
    if not path:
        return []
    if not os.path.exists(path):
        return []
    phrases: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                phrases.append(s)
    return phrases


# ----------------------------- FFmpeg 处理 -----------------------------

class RtspSegmenter:
    def __init__(
        self,
        rtsp_url: str,
        out_dir_video: str,
        segment_time: int = 10,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.out_dir_video = out_dir_video
        self.segment_time = int(segment_time)
        self.proc: Optional[subprocess.Popen] = None
        ensure_dir(out_dir_video)

    def build_cmd(self) -> List[str]:
        # 输出文件名：seg_YYYYmmdd_HHMMSS.mp4
        pattern = os.path.join(self.out_dir_video, "seg_%Y%m%d_%H%M%S.mp4")
        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "5000000",  # 5s
            "-i",
            self.rtsp_url,
            # 保留视频原编码，音频转为 aac 以保证 mp4 兼容，避免部分摄像头的 G.711 无法封装
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            # 分段参数
            "-f",
            "segment",
            "-reset_timestamps",
            "1",
            "-segment_time",
            str(self.segment_time),
            "-strftime",
            "1",
            pattern,
        ]
        return cmd

    def start(self) -> None:
        self.stop()
        cmd = self.build_cmd()
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            finally:
                self.proc = None


def extract_wav_from_mp4(mp4_path: str, wav_path: str, timeout: int = 15) -> bool:
    """从 mp4 提取 16k 单声道 wav。"""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        mp4_path,
        "-vn",
        "-sn",
        "-dn",
        "-map",
        "0:a:0?",  # 显式映射首个音频流（可选）
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "aresample=async=1:min_hard_comp=0.100:first_pts=0",
        "-acodec",
        "pcm_s16le",
        wav_path,
    ]
    try:
        ret = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout
        )
        ok = ret.returncode == 0 and os.path.exists(wav_path)
        if not ok:
            # 打印简要诊断信息
            msg = ret.stderr.decode("utf-8", errors="ignore").strip()
            if msg:
                print(f"[monitor] ffmpeg 提取失败: {os.path.basename(mp4_path)} => {msg.splitlines()[-1]}")
        return ok
    except subprocess.TimeoutExpired:
        print(f"[monitor] ffmpeg 提取超时: {os.path.basename(mp4_path)}")
        return False
    except Exception as e:
        print(f"[monitor] 提取异常: {e}")
        return False


def ensure_wav_16k_mono(src_audio: str, dst_wav: str, timeout: int = 15) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_audio,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        dst_wav,
    ]
    try:
        ret = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=timeout
        )
        return ret.returncode == 0 and os.path.exists(dst_wav)
    except subprocess.TimeoutExpired:
        print(f"[monitor] 参考音频转换超时: {os.path.basename(src_audio)}")
        return False
    except Exception:
        return False


def concat_wav(sys_wav: str, seg_wav: str, out_wav: str, timeout: int = 15) -> bool:
    """将系统参考音频与片段音频首尾拼接为 out_wav。"""
    # 使用 concat filter，保证音频参数一致
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        sys_wav,
        "-i",
        sys_wav,
        "-i",
        seg_wav,
        "-filter_complex",
        "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]",
        "-map",
        "[a]",
        out_wav,
    ]
    try:
        ret = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=timeout
        )
        return ret.returncode == 0 and os.path.exists(out_wav)
    except subprocess.TimeoutExpired:
        print(f"[monitor] 音频拼接超时: {os.path.basename(seg_wav)}")
        return False
    except Exception:
        return False


# ----------------------------- 告警上报 -----------------------------

def post_alert(safety_url: str, when: str, timeout: int = 2):
    """按协议 POST 上报，
    返回 (http_status, body_code, body_msg)
    """
    payload = {
		"deviceId":"1287",
		"deviceIp":"127.0.0.1",
		"safetyId":"1",
		"safetyName":"安全员1",
		"warning":"way",
		"type":1,
		"safetyUrl":safety_url,
		"brakeUrl":"",
		"datatime":when,
		"imgUrl":""
		}
    url = "http://182.92.230.1:9181/warning-agreement/upload"

    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    body_code = None
    body_msg = None
    try:
        j = r.json()
        body_code = int(j.get("code")) if "code" in j else None
        body_msg = str(j.get("msg")) if "msg" in j else None
    except Exception:
        pass
    return r.status_code, body_code, body_msg


# ----------------------------- 监控主逻辑 -----------------------------

class Monitor:
    def __init__(
        self,
        rtsp_url: str,
        system_audio: str,
        out_dir: str = "/data/alarms",
        chunk_seconds: int = 10,
        target: str = "BM1684X",
        dev_id: int = 0,
        system_phrases_path: Optional[str] = None
    ) -> None:
        self.rtsp_url = rtsp_url
        self.system_audio = system_audio
        self.out_dir = out_dir
        self.chunk_seconds = int(chunk_seconds)
        self.target = target
        self.dev_id = dev_id
        self.system_phrases = load_system_phrases(system_phrases_path)

        self.dir_video = os.path.join(self.out_dir, "video")
        self.dir_audio_tmp = os.path.join(self.out_dir, "audio_tmp")
        ensure_dir(self.dir_video)
        ensure_dir(self.dir_audio_tmp)

        self.segmenter = RtspSegmenter(self.rtsp_url, self.dir_video, self.chunk_seconds)
        self.stop_event = threading.Event()
        self.processed_files: set[str] = set()
        self.ffmpeg_timeout = 15  # seconds for ffmpeg operations

        # 上报异步队列
        self._alert_q: "queue.Queue[Tuple[str, str, List[tuple]]]" = queue.Queue()
        self._alert_thread = threading.Thread(target=self._alert_worker, name="alert-worker", daemon=True)

        # 预处理系统参考音频到 16k 单声道 wav
        self.sys_wav_16k = os.path.join(self.dir_audio_tmp, "system_ref_16k.wav")
        if not ensure_wav_16k_mono(self.system_audio, self.sys_wav_16k, timeout=self.ffmpeg_timeout):
            raise RuntimeError(f"系统参考音频无法转换为 16k wav: {self.system_audio}")

        # 初始化 FunASR 模型（与 offline.py 保持一致）
        self.model = AutoModel(
            model=f"speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404/models/{self.target}",
            vad_model=f"speech_fsmn_vad_zh-cn-16k-common/models/{self.target}",
            punc_model=f"punc_ct-transformer_zh-cn-common-vocab272727/models/{self.target}",
            spk_model=f"speech_campplus_sv_zh-cn_16k-common/models/{self.target}",
            device="cpu",
            disable_update=True,
            disable_pbar=True,
            dev_id=self.dev_id,
        )

        self.ip = get_local_ip_fallback()
        # 存储限制配置
        self.max_bytes = 1024 * 1024 * 1024  # 1G
        self.max_age_sec = 24 * 3600  # 1天
        self._last_cleanup_ts = 0.0
        # 启动异步上报线程
        self._alert_thread.start()

    # --------- 匹配系统提示词 ---------
    def is_system_phrase(self, text: str) -> bool:
        """按字符数匹配系统提示词：与库中任意一句的最长公共子串长度 >= 5 视为命中。"""
        if not self.system_phrases:
            return False
        for p in self.system_phrases:
            if longest_common_substring_len(text, p) >= 5:
                return True
        return False

    # --------- 主循环 ---------
    def run(self) -> None:
        # 启动分段
        self.segmenter.start()
        print("[monitor] RTSP 分段已启动")

        try:
            while not self.stop_event.is_set():
                # 保活 ffmpeg
                if not self.segmenter.is_running():
                    print("[monitor] ffmpeg 已退出，尝试重启...")
                    self.segmenter.start()

                # 扫描新的视频段
                try:
                    files = sorted(
                        [
                            f
                            for f in os.listdir(self.dir_video)
                            if f.startswith("seg_") and f.endswith(".mp4")
                        ]
                    )
                except FileNotFoundError:
                    files = []

                now = time.time()
                for fname in files:
                    if fname in self.processed_files:
                        continue
                    fpath = os.path.join(self.dir_video, fname)
                    # 避免处理正在写入的文件：至少间隔 1s
                    try:
                        mtime = os.path.getmtime(fpath)
                    except FileNotFoundError:
                        continue
                    if now - mtime < 1.0 or not self._is_file_stable(fpath):
                        continue

                    # 处理该切片
                    try:
                        self._process_segment(fpath)
                    except Exception as e:
                        print(f"[monitor] 处理失败 {fname}: {e}")
                    finally:
                        self.processed_files.add(fname)

                # processed_files 防膨胀
                if len(self.processed_files) > 10000:
                    self.processed_files.clear()

                # 定期清理：空间与时效
                if time.time() - self._last_cleanup_ts > 30:
                    try:
                        self._enforce_retention_and_quota()
                    except Exception as e:
                        print(f"[monitor] 清理异常: {e}")
                    self._last_cleanup_ts = time.time()

                time.sleep(0.2)
        finally:
            self.segmenter.stop()
            # 通知并等待上报线程结束
            self._stop_alert_worker()

    def _process_segment(self, mp4_path: str) -> None:
        base = os.path.splitext(os.path.basename(mp4_path))[0]
        wav_path = os.path.join(self.dir_audio_tmp, base + ".wav")
        joined_wav = os.path.join(self.dir_audio_tmp, base + "_joined.wav")

        if not extract_wav_from_mp4(mp4_path, wav_path, timeout=self.ffmpeg_timeout):
            print(f"[monitor] 提取音频失败: {mp4_path}")
            return

        if not concat_wav(self.sys_wav_16k, wav_path, joined_wav, timeout=self.ffmpeg_timeout):
            print(f"[monitor] 拼接音频失败: {wav_path}")
            return

        # 识别
        res = self.model.generate(input=joined_wav, batch_size_s=self.chunk_seconds)
        suspicious = False
        details = []

        # 新判定逻辑：
        # - 若 spk_id 不全为 0 => 直接上报
        # - 若全为 0 => 文本判定（是否全是系统提示词）
        if isinstance(res, list) and res and isinstance(res[0], dict):
            item = res[0]
            if "sentence_info" in item and isinstance(item["sentence_info"], list) and item["sentence_info"]:
                spks = []
                texts = []
                for si in item["sentence_info"]:
                    spk = int(si.get("spk", -1))
                    text = str(si.get("text", ""))
                    details.append((spk, text))
                    spks.append(spk)
                    texts.append(text)
                all_zero = all(s == 0 for s in spks)
                if not all_zero:
                    suspicious = True
                else:
                    # 全 0 则检查文本是否全部为系统提示词
                    suspicious = not all(self.is_system_phrase(t) for t in texts if t)
            else:
                # 无句级信息，回退到纯文本判定
                text = str(item.get("text", ""))
                details.append((0, text))
                suspicious = not (self.is_system_phrase(text) if text else False)

        print(f"[monitor] {os.path.basename(mp4_path)} 识别结果: {details}")

        if suspicious:
            # 留存视频文件（文件已在 out_dir/video 下，无需再复制；直接构造 URL）
            rel_path = os.path.relpath(mp4_path, self.out_dir)
            # 如果不在 out_dir 下，移动到 out_dir/video
            if rel_path.startswith(".."):
                dst = os.path.join(self.dir_video, os.path.basename(mp4_path))
                try:
                    shutil.move(mp4_path, dst)
                    rel_path = os.path.relpath(dst, self.out_dir)
                except Exception:
                    # 复制失败也不影响上报
                    rel_path = os.path.basename(mp4_path)

            safety_url = f"http://{self.ip}/static/{rel_path}"
            # 从文件名提取时间，否则用当前时间
            ts = self._datetime_from_filename(os.path.basename(mp4_path)) or dt.datetime.now()
            datatime = ts.strftime("%Y-%m-%d %H:%M:%S")

            print(f"[monitor] 触发告警 -> {safety_url} @ {datatime}，已入队上报")
            self._enqueue_alert(safety_url, datatime, details)

        # 清理中间音频
        for p in (wav_path, joined_wav):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        # 未触发告警则删除视频，避免占用空间
        if not suspicious:
            try:
                if os.path.exists(mp4_path):
                    os.remove(mp4_path)
            except Exception:
                pass

    def _is_file_stable(self, path: str, checks: int = 2, interval: float = 0.3) -> bool:
        """检测文件大小在短时间内是否稳定，避免处理未完全写入的分段。"""
        try:
            last = os.path.getsize(path)
            for _ in range(max(1, checks)):
                time.sleep(max(0.05, interval))
                cur = os.path.getsize(path)
                if cur != last:
                    last = cur
                    return False
            return True
        except Exception:
            return False

    def _datetime_from_filename(self, fname: str) -> Optional[dt.datetime]:
        # 形如 seg_YYYYmmdd_HHMMSS.mp4
        m = re.match(r"seg_(\d{8})_(\d{6})", fname)
        if not m:
            return None
        try:
            return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            return None

    def _append_alert_log(self, url: str, when: str, details: List[tuple], http_status: Optional[int], body_code: Optional[int], body_msg: Optional[str]) -> None:
        log_dir = os.path.join(os.path.dirname(__file__), "alerts")
        ensure_dir(log_dir)
        log_path = os.path.join(log_dir, "alerts.log")
        line = json.dumps({
            "time": when,
            "url": url,
            "details": details,
            "http_status": http_status,
            "code": body_code,
            "msg": body_msg,
        }, ensure_ascii=False)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _enforce_retention_and_quota(self) -> None:
        """删除超期文件，并按时间从旧到新删除以满足空间上限。"""
        # 1) 删除超过保存期的文件
        now = time.time()
        video_files = []
        try:
            for fname in os.listdir(self.dir_video):
                if not fname.endswith(".mp4"):
                    continue
                fpath = os.path.join(self.dir_video, fname)
                try:
                    st = os.stat(fpath)
                except FileNotFoundError:
                    continue
                # 删除超龄
                if now - st.st_mtime > self.max_age_sec:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                else:
                    video_files.append((fpath, st.st_mtime, st.st_size))
        except FileNotFoundError:
            pass

        # 2) 若空间超限，按时间从旧到新删除
        total = sum(s for _, _, s in video_files)
        if total > self.max_bytes:
            # 按修改时间升序
            video_files.sort(key=lambda x: x[1])
            idx = 0
            while total > self.max_bytes and idx < len(video_files):
                fpath, _, sz = video_files[idx]
                try:
                    os.remove(fpath)
                    total -= sz
                except Exception:
                    pass
                idx += 1

    # ---------- 上报异步处理 ----------
    def _enqueue_alert(self, url: str, when: str, details: List[tuple]) -> None:
        try:
            self._alert_q.put_nowait((url, when, details))
        except Exception:
            # 队列异常时直接同步上报作为兜底
            http_status, body_code, body_msg = post_alert(url, when, timeout=2)
            if body_code is not None:
                print(f"[monitor] 上报返回 code={body_code}, msg={body_msg}")
            else:
                print(f"[monitor] 上报返回 http_status={http_status}")
            self._append_alert_log(url, when, details, http_status, body_code, body_msg)

    def _alert_worker(self) -> None:
        while not self.stop_event.is_set() or not self._alert_q.empty():
            try:
                url, when, details = self._alert_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                http_status, body_code, body_msg = post_alert(url, when, timeout=2)
                if body_code is not None:
                    print(f"[monitor] 上报返回 code={body_code}, msg={body_msg}")
                else:
                    print(f"[monitor] 上报返回 http_status={http_status}")
                self._append_alert_log(url, when, details, http_status, body_code, body_msg)
            except Exception as e:
                print(f"[monitor] 上报异常: {e}")
            finally:
                try:
                    self._alert_q.task_done()
                except Exception:
                    pass

    def _stop_alert_worker(self) -> None:
        # 等待队列尽量清空
        deadline = time.time() + 2.5
        while not self._alert_q.empty() and time.time() < deadline:
            time.sleep(0.1)
        # 线程将根据 stop_event 退出
        try:
            if self._alert_thread.is_alive():
                self._alert_thread.join(timeout=2)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监控场景下非系统人声检测与上报")
    parser.add_argument("--rtsp", required=True, help="RTSP 地址，例如 rtsp://user:pass@ip:554/Streaming/Channels/101")
    parser.add_argument(
        "--system-audio",
        default="../system_ref.wav",
        help="系统参考音频，作为前置拼接，系统说话人 spk=0",
    )
    parser.add_argument("--out-dir", default="/data/alarms", help="告警本地保存目录 (已由 nginx 映射到 /static)")
    parser.add_argument("--chunk-seconds", type=int, default=10, help="检测窗口秒数，默认 10s")
    parser.add_argument("--target", default="BM1684X", help="模型目标平台目录名，如 BM1684X/BM1688")
    parser.add_argument("--dev-id", type=int, default=0, help="设备 ID")
    parser.add_argument(
        "--system-phrases",
        default="../system_phrases.txt",
        help="系统提示词文本库（每行一条），用于过滤系统播报文本",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    monitor = Monitor(
        rtsp_url=args.rtsp,
        system_audio=args.system_audio,
        out_dir=args.out_dir,
        chunk_seconds=args.chunk_seconds,
        target=args.target,
        dev_id=args.dev_id,
        system_phrases_path=args.system_phrases
    )

    # 优雅退出
    def handle_sig(signum, frame):
        print("[monitor] 收到退出信号，正在停止...")
        monitor.stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    monitor.run()


if __name__ == "__main__":
    main()
