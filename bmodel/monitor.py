#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
驾考车内语音检测监控：
- 输入：RTSP 音视频流（仅取音频）或本地 WAV 文件
- 每 10s 离线检测一次，静音段跳过 ASR
- ASR 采用本地 FunASR bmodel 推理（参考 offline.py 配置）
- 区分系统/非系统声音：
	1) 使用 spk 区分（默认 system_spk_ids={0}）
	2) 多说话人时，结合识别文本和 system_words 进一步判别
- 对非系统声音进行日志报警并保存该 10s 片段
"""

import argparse
import difflib
import logging
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# FunASR 引入
sys.path.append("..")
from funasr import AutoModel  # noqa: E402

# 配置：系统提示词库（可按需增删）
system_words = [
	"信号灯", "行驶", "照明", "道路",
	"夜间通过急弯、坡路、拱桥、人行道或没有交通信号灯路口",
	"在无照明的道路行驶",
	"在有信号灯控制路口转弯",
	"在照明不良的道路行驶",
	"在照明良好的道路上行驶",
]

# （示例 RTSP 常量已移除）


# ===================== 基础工具 =====================

def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def setup_logger(log_dir: Path) -> logging.Logger:
	ensure_dir(log_dir)
	logger = logging.getLogger("monitor")
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter(
		fmt="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	# 控制台
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.INFO)
	ch.setFormatter(fmt)
	logger.addHandler(ch)
	# 文件
	fh = logging.FileHandler(log_dir / "alerts.log", encoding="utf-8")
	fh.setLevel(logging.INFO)
	fh.setFormatter(fmt)
	logger.addHandler(fh)
	return logger

def load_wav(path: Path) -> Tuple[np.ndarray, int]:
	with wave.open(str(path), "rb") as wf:
		n_channels = wf.getnchannels()
		sampwidth = wf.getsampwidth()
		framerate = wf.getframerate()
		n_frames = wf.getnframes()
		raw = wf.readframes(n_frames)
	if sampwidth != 2:
		raise ValueError(f"仅支持 16-bit PCM，got sampwidth={sampwidth}")
	data = np.frombuffer(raw, dtype=np.int16)
	if n_channels > 1:
		data = data.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
	return data.astype(np.float32) / 32768.0, framerate


def write_wav(path: Path, pcm: np.ndarray, sr: int = 16000):
	pcm_int16 = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
	with wave.open(str(path), "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sr)
		wf.writeframes(pcm_int16.tobytes())


def rms_energy(x: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
	if len(x) < frame:
		return np.array([], dtype=np.float32)
	n = 1 + (len(x) - frame) // hop
	e = np.empty(n, dtype=np.float32)
	for i in range(n):
		seg = x[i * hop : i * hop + frame]
		e[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
	return e


def _ensure_mono_16k(x: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
	"""线性插值重采样到 16k 单声道。输入已是单通道 float32。"""
	if sr == 16000:
		return x.astype(np.float32), 16000
	t_old = np.linspace(0, len(x) / sr, num=len(x), endpoint=False)
	t_new = np.linspace(0, len(x) / sr, num=int(len(x) * 16000 / sr), endpoint=False)
	x16 = np.interp(t_new, t_old, x).astype(np.float32)
	return x16, 16000


def is_mostly_silence(
	wav_path: Path,
	energy_th: float = 0.01,
	min_voiced_ratio: float = 0.05,
) -> bool:
	try:
		x, sr = load_wav(wav_path)
		if sr != 16000:
			x, sr = _ensure_mono_16k(x, sr)
		e = rms_energy(x)
		if e.size == 0:
			return True
		voiced = (e > energy_th).sum()
		ratio = voiced / float(len(e))
		return ratio < min_voiced_ratio
	except Exception:
		return False


# ===================== 分段器 =====================

def start_ffmpeg_segmenter(
	rtsp_url: str,
	out_dir: Path,
	segment_seconds: int = 10,
	sample_rate: int = 16000,
) -> subprocess.Popen:
	ensure_dir(out_dir)
	# 使用 segment 切分 10s wav；-reset_timestamps 1 便于按文件名排序
	out_tpl = str(out_dir / "%Y%m%d_%H%M%S_%03d.wav")
	cmd = [
		"ffmpeg",
		"-rtsp_transport",
		"tcp",
		"-i",
		rtsp_url,
		"-vn",
		"-acodec",
		"pcm_s16le",
		"-ar",
		str(sample_rate),
		"-ac",
		"1",
		"-f",
		"segment",
		"-segment_time",
		str(segment_seconds),
		"-reset_timestamps",
		"1",
		"-nostdin",
		"-hide_banner",
		"-loglevel",
		"error",
		out_tpl,
	]
	return subprocess.Popen(cmd)


def chunk_local_wav(
	wav_path: Path,
	out_dir: Path,
	segment_seconds: int = 10,
) -> None:
	"""将本地 wav 切分为 10s 片段到 out_dir。采样率保持原文件。"""
	ensure_dir(out_dir)
	x, sr = load_wav(wav_path)
	seg_len = segment_seconds * sr
	total = len(x)
	t = int(time.time())
	idx = 0
	for start in range(0, total, seg_len):
		end = min(start + seg_len, total)
		seg = x[start:end]
		fn = out_dir / time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
		fn = Path(f"{fn}_{idx:03d}.wav")
		write_wav(fn, seg, sr)
		idx += 1


# ===================== 判别与 ASR =====================

_PUNCS_RE = re.compile(r"[，。、“”‘’？：；！,.!?;:\-\(\)\[\]{}…·~`]+")


def normalize_text(s: str) -> str:
	s = _PUNCS_RE.sub("", s)
	s = re.sub(r"\s+", "", s)
	return s.lower()


def _char_coverage(a: str, b: str) -> float:
	if not a or not b:
		return 0.0
	sa, sb = set(a), set(b)
	if not sb:
		return 0.0
	return len(sa & sb) / float(len(sb))


def _ratio(a: str, b: str) -> float:
	if not a or not b:
		return 0.0
	return difflib.SequenceMatcher(None, a, b).ratio()


def _partial_ratio(a: str, b: str) -> float:
	# 取较短串在较长串中的最佳窗口相似度
	if not a or not b:
		return 0.0
	short, long = (a, b) if len(a) <= len(b) else (b, a)
	m, n = len(short), len(long)
	if m == 0 or n == 0:
		return 0.0
	if m >= n:
		return _ratio(short, long)
	best = 0.0
	# 步进为1，短串通常较短，性能可接受
	for i in range(0, n - m + 1):
		r = _ratio(short, long[i : i + m])
		if r > best:
			best = r
			if best >= 0.99:
				break
	return best

def _ngram_set(s: str, n: int) -> set:
	if len(s) < n:
		return set()
	return {s[i:i+n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
	if not a or not b:
		return 0.0
	inter = len(a & b)
	union = len(a | b)
	return inter / float(union) if union else 0.0


def text_matches_system(text: str, system_lib: List[str]) -> bool:
	nt = normalize_text(text)
	if not nt:
		return False
	for w in system_lib:
		kw = normalize_text(w)
		if not kw:
			continue
		# 关键词包含：识别文本包含关键词即视为系统文本
		if kw in nt:
			return True
	return False


@dataclass
class ASRConfig:
	target: str = "BM1684X"
	dev_id: int = 0
	device: str = "cpu"
	disable_update: bool = True
	disable_pbar: bool = True
	batch_size_s: int = 10
	system_spk_ids: Tuple[int, ...] = (0,)  # 默认 spk=0 为系统


class ASRWorker:
	def __init__(self, cfg: ASRConfig, alerts_dir: Path, logger: logging.Logger, system_ref_path: Optional[Path] = None):
		self.cfg = cfg
		self.alerts_dir = alerts_dir
		self.logger = logger
		target = cfg.target
		self.model = AutoModel(
			model=f"speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404/models/{target}",
			vad_model=f"speech_fsmn_vad_zh-cn-16k-common/models/{target}",
			punc_model=f"punc_ct-transformer_zh-cn-common-vocab272727/models/{target}",
			spk_model=f"speech_campplus_sv_zh-cn_16k-common/models/{target}",
			device=cfg.device,
			disable_update=cfg.disable_update,
			disable_pbar=cfg.disable_pbar,
			dev_id=cfg.dev_id,
		)
		# 预加载系统参考音
		self.system_ref_pcm: Optional[np.ndarray] = None
		if system_ref_path and system_ref_path.exists():
			try:
				pcm, sr = load_wav(system_ref_path)
				pcm, _ = _ensure_mono_16k(pcm, sr)
				# 限制参考音长度（最多 5 秒）
				max_len = 5 * 16000
				if len(pcm) > max_len:
					pcm = pcm[:max_len]
				self.system_ref_pcm = pcm
				self.logger.info(f"loaded system_ref: {system_ref_path.name}, {len(pcm)/16000:.2f}s")
			except Exception as e:
				self.logger.warning(f"failed to load system_ref '{system_ref_path}': {e}")

	def infer(self, wav_path: Path) -> Dict:
		# 若存在系统参考音，则拼接后送入 ASR
		if self.system_ref_pcm is not None:
			try:
				seg_pcm, sr = load_wav(wav_path)
				seg_pcm, _ = _ensure_mono_16k(seg_pcm, sr)
				ref = self.system_ref_pcm
				ref_len = len(ref)
				seg_len = len(seg_pcm)
				min_gap = 16000  # 1s
				required_total = 320000  # 20s
				parts = [ref, np.zeros(min_gap, dtype=np.float32), seg_pcm, np.zeros(min_gap, dtype=np.float32)]
				total_len = ref_len + 2*min_gap + seg_len
				if total_len < required_total:
					parts.extend([seg_pcm[:required_total - total_len]])
				cat = np.concatenate(parts)
				with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
					tmp_path = Path(tf.name)
				write_wav(tmp_path, cat, 16000)
				try:
					res = self.model.generate(input=str(tmp_path), batch_size_s=self.cfg.batch_size_s)
				finally:
					try:
						tmp_path.unlink(missing_ok=True)
					except Exception:
						pass
				return res[0]
			except Exception as e:
				self.logger.warning(f"concat system_ref failed, fallback to raw segment: {e}")
		res = self.model.generate(input=str(wav_path), batch_size_s=self.cfg.batch_size_s)
		return res[0]

	def classify_segment(self, asr_out: Dict) -> Tuple[bool, str, str]:
		"""返回: (is_non_system, reason, summary_text)
		规则：
		- 使用参考音时，跳过第一个句子的判断；
		- 若任一句子的 spk 不在 system_spk_ids，直接报警；
		- 若无句级信息，回退整段文本匹配。
		"""
		summary_text = asr_out.get("text", "") or ""
		sent_info = asr_out.get("sentence_info", []) or []
		skip_first = self.system_ref_pcm is not None

		# 为日志/告警提供“去掉参考音”的文本视图
		def _effective_text() -> str:
			if sent_info:
				start = 1 if skip_first else 0
				parts = []
				for idx, si in enumerate(sent_info):
					if idx < start:
						continue
					parts.append((si.get("text", "") or "").strip())
				return " ".join([p for p in parts if p])
			return summary_text

		if sent_info:
			# 1) spk 检查
			for idx, si in enumerate(sent_info):
				if skip_first and idx == 0:
					continue
				spk = int(si.get("spk", -1))
				if spk not in self.cfg.system_spk_ids:
					return True, f"non-system spk found (spk={spk}, idx={idx})", _effective_text()
			# # 2) 文本检查
			# for idx, si in enumerate(sent_info):
			# 	if skip_first and idx == 0:
			# 		continue
			# 	text = (si.get("text", "") or "").strip()
			# 	if text and not text_matches_system(text, system_words):
			# 		return True, f"text not in system_words (idx={idx})", _effective_text()
			return False, "system voice", _effective_text()

		# 无句级信息：回退整段文本
		if summary_text and not text_matches_system(summary_text, system_words):
			return True, "text not in system_words (no sentence_info)", summary_text
		return False, "system voice", summary_text

	def save_alert(self, wav_path: Path, reason: str, summary: str) -> Path:
		ensure_dir(self.alerts_dir)
		ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
		safe_reason = re.sub(r"[^\w\u4e00-\u9fff]+", "_", reason).strip("_")[:40]
		dst = self.alerts_dir / f"alert_{ts}_{safe_reason}.wav"
		# 移动文件避免多余的读写复制
		try:
			shutil.move(str(wav_path), str(dst))
		except Exception:
			# 若跨设备移动失败则回退为拷贝
			shutil.copy2(wav_path, dst)
		# 记录文本
		with open(self.alerts_dir / f"alert_{ts}.txt", "w", encoding="utf-8") as f:
			f.write(f"reason: {reason}\ntext: {summary}\nsource: {wav_path}\n")
		return dst


# ===================== 线程管道 =====================

def file_watcher(dir_path: Path, q_out: queue.Queue, stop_event: threading.Event, logger: logging.Logger):
	seen = set()
	logger.info(f"watching directory: {dir_path}")
	while not stop_event.is_set():
		try:
			for p in sorted(dir_path.glob("*.wav")):
				if p in seen:
					continue
				# 确保文件写入完成（修改时间超过 0.5s）
				if time.time() - p.stat().st_mtime < 0.5:
					continue
				seen.add(p)
				q_out.put(p)
		except Exception as e:
			logger.warning(f"watcher error: {e}")
		time.sleep(0.5)


def vad_filter_worker(q_in: queue.Queue, q_out: queue.Queue, stop_event: threading.Event, logger: logging.Logger):
	while not stop_event.is_set():
		try:
			p: Path = q_in.get(timeout=0.5)
		except queue.Empty:
			continue
		try:
			if is_mostly_silence(p):
				logger.info(f"skip silent segment: {p.name}")
				try:
					p.unlink(missing_ok=True)
				except Exception:
					pass
			else:
				q_out.put(p)
		except Exception as e:
			logger.error(f"VAD error on {p}: {e}")
		finally:
			q_in.task_done()


def asr_consumer(q_in: queue.Queue, stop_event: threading.Event, worker: ASRWorker, logger: logging.Logger, delete_after: bool = True):
	while not stop_event.is_set():
		try:
			p: Path = q_in.get(timeout=0.5)
		except queue.Empty:
			continue
		try:
			asr_out = worker.infer(p)
			is_abn, reason, summary = worker.classify_segment(asr_out)
			if is_abn:
				dst = worker.save_alert(p, reason, summary)
				logger.warning(f"ALERT: {reason}; text='{summary}'; saved={dst.name}")
			else:
				logger.info(f"system OK; text='{summary}'")
		except Exception as e:
			logger.error(f"ASR error on {p}: {e}")
		finally:
			if delete_after:
				try:
					p.unlink(missing_ok=True)
				except Exception:
					pass
			q_in.task_done()


# ===================== 主流程 =====================

def main():
	parser = argparse.ArgumentParser(description="驾考车语音监控")
	g = parser.add_mutually_exclusive_group(required=True)
	g.add_argument("--rtsp", type=str, help="RTSP URL")
	g.add_argument("--input", type=str, help="本地 WAV 文件用于调试")
	parser.add_argument("--segment", type=int, default=10, help="分段秒数，默认 10")
	parser.add_argument("--out", type=str, default="segments", help="临时分段输出目录")
	parser.add_argument("--alerts", type=str, default="alerts", help="告警保存目录")
	parser.add_argument("--workers", type=int, default=1, help="ASR 并发数")
	parser.add_argument("--target", type=str, default="BM1684X", help="bmodel 目标后端")
	parser.add_argument("--system-spk", type=int, nargs="+", default=[0], help="系统声音 spk id 列表，如 0 或 0 1")
	parser.add_argument("--system_ref", type=str, default=None, help="系统参考音 WAV（前置到每个片段以稳定 spk=0 为系统）")
	args = parser.parse_args()

	base_dir = Path(__file__).parent
	seg_dir = (base_dir / args.out).resolve()
	alerts_dir = (base_dir / args.alerts).resolve()
	logger = setup_logger(alerts_dir)

	logger.info("starting monitor")
	logger.info(f"mode={'RTSP' if args.rtsp else 'FILE'} segment={args.segment}s workers={args.workers}")

	ensure_dir(seg_dir)
	ensure_dir(alerts_dir)

	stop_event = threading.Event()

	# 优雅退出
	def _sig_handler(signum, frame):
		logger.info("stopping...")
		stop_event.set()

	signal.signal(signal.SIGINT, _sig_handler)
	signal.signal(signal.SIGTERM, _sig_handler)

	# 分段源：RTSP 或 本地文件
	ffmpeg_proc: Optional[subprocess.Popen] = None
	try:
		if args.rtsp:
			logger.info(f"start ffmpeg segmenter from {args.rtsp}")
			ffmpeg_proc = start_ffmpeg_segmenter(args.rtsp, seg_dir, args.segment)
		else:
			src = Path(args.input).resolve()
			logger.info(f"chunk local wav: {src}")
			chunk_local_wav(src, seg_dir, args.segment)
	except FileNotFoundError:
		logger.error("无法启动 ffmpeg，请确保已安装并可执行: 'ffmpeg'")
		return
	except Exception as e:
		logger.error(f"启动分段器失败: {e}")
		return

	# 队列流水线
	q_files: queue.Queue = queue.Queue(maxsize=64)
	q_voiced: queue.Queue = queue.Queue(maxsize=64)

	watcher = threading.Thread(target=file_watcher, name="watcher", args=(seg_dir, q_files, stop_event, logger), daemon=True)
	vad_worker = threading.Thread(target=vad_filter_worker, name="vad", args=(q_files, q_voiced, stop_event, logger), daemon=True)
	watcher.start()
	vad_worker.start()

	# ASR 工作线程
	asr_threads: List[threading.Thread] = []
	for i in range(max(1, args.workers)):
		worker = ASRWorker(
			ASRConfig(target=args.target, system_spk_ids=tuple(args.system_spk)),
			alerts_dir,
			logger,
			Path(args.system_ref).resolve() if args.system_ref else None,
		)
		t = threading.Thread(target=asr_consumer, name=f"asr-{i}", args=(q_voiced, stop_event, worker, logger))
		t.daemon = True
		t.start()
		asr_threads.append(t)

	# 主循环：仅在文件模式下等待处理完成后退出；RTSP 模式常驻
	try:
		if args.rtsp:
			while not stop_event.is_set():
				time.sleep(1.0)
		else:
			# 等待所有切分片处理完
			while not stop_event.is_set():
				if q_files.empty() and q_voiced.empty() and not any(seg_dir.glob("*.wav")):
					break
				time.sleep(0.5)
	finally:
		stop_event.set()
		# 清理 ffmpeg 进程
		if ffmpeg_proc is not None:
			try:
				ffmpeg_proc.terminate()
			except Exception:
				pass
		logger.info("monitor stopped")


if __name__ == "__main__":
	main()


