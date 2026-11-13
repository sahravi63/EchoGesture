"""EchoGesture live captions: webcam (or a video file) -> on-screen caption HUD.

    python src/demo/live.py                    # default webcam
    python src/demo/live.py --camera 1         # another camera
    python src/demo/live.py --video clip.mp4   # play a file through the same pipeline
    python src/demo/live.py --tts              # also speak each word (offline, pyttsx3)

Everything runs on this machine: frames go to MediaPipe, only landmarks are kept, the
INT8 ensemble classifies each sign, and the word is drawn on screen (and optionally
spoken). Nothing is uploaded and no frame is saved unless you pass --record.

Keys: q / Esc quit · space end the current sign now · c clear transcript ·
      t toggle speech · l toggle hand skeleton

How signs are found (no button needed): a sign starts when a hand is visible for a
couple of frames, and ends when the hands drop out of view or stay still for ~0.5 s
(or after 4 s). The buffered landmarks of that sign go through predict()'s exact
feature pipeline and models.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "demo"))

import hud  # noqa: E402


class SignSegmenter:
    """Cuts a stream of landmark rows [543,3] into signs."""

    def __init__(self, fps=15.0, start_frames=2, end_s=0.5, still_s=0.6, max_s=4.0,
                 min_frames=6, preroll=3, still_thresh=0.004):
        self.fps = fps
        self.start_frames, self.min_frames, self.preroll = start_frames, min_frames, preroll
        self.end_n = max(2, int(round(end_s * fps)))
        self.still_n = max(3, int(round(still_s * fps)))
        self.max_n = int(round(max_s * fps))
        self.still_thresh = still_thresh
        self.reset()

    def reset(self):
        self.buf, self.pre = [], []
        self.active = False
        self.seen = self.gone = self.still = 0
        self.moved = False
        self.last_wrist = None

    @staticmethod
    def _wrists(row):
        w = row[[468, 522], :2]                          # left / right wrist
        return w[~np.isnan(w[:, 0])]

    def feed(self, row):
        """Add one frame of landmarks. Returns a finished sign [T,543,3] or None."""
        wr = self._wrists(row)
        has_hand = len(wr) > 0
        if not self.active:
            self.pre = (self.pre + [row])[-self.preroll:]
            self.seen = self.seen + 1 if has_hand else 0
            if self.seen >= self.start_frames:
                self.active, self.buf = True, list(self.pre)
                self.gone = self.still = 0
                self.moved, self.last_wrist = False, None
            return None
        self.buf.append(row)
        self.gone = 0 if has_hand else self.gone + 1
        if has_hand:
            m = wr.mean(0)
            if self.last_wrist is not None:
                step = float(np.linalg.norm(m - self.last_wrist))
                if step > 3 * self.still_thresh:
                    self.moved, self.still = True, 0
                elif step < self.still_thresh:
                    self.still += 1
            self.last_wrist = m
        done = (self.gone >= self.end_n or (self.moved and self.still >= self.still_n)
                or len(self.buf) >= self.max_n)
        return self.flush() if done else None

    def flush(self):
        """End the current sign now (space bar). Returns it, or None if too short."""
        sign = self.buf
        self.reset()
        if len(sign) < self.min_frames:
            return None
        return np.stack(sign)


class Speaker:
    """Offline text-to-speech in a background thread (pyttsx3 if installed)."""

    def __init__(self, enabled):
        self.q = queue.Queue()
        self.ok = self.available = False
        if enabled:
            try:
                import pyttsx3                            # noqa: F401
                self.ok = self.available = True
                threading.Thread(target=self._run, daemon=True).start()
            except Exception as e:
                print(f"[speech off] pyttsx3 not available: {e}")

    def _run(self):
        import pyttsx3
        eng = pyttsx3.init()
        while True:
            eng.say(self.q.get())
            eng.runAndWait()

    def say(self, word):
        if self.ok:
            self.q.put(word.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="play a video file instead of the webcam")
    ap.add_argument("--tts", action="store_true", help="speak each recognised word")
    ap.add_argument("--min-conf", type=float, default=0.30, help="below this, show '?' instead of a word")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam preview")
    ap.add_argument("--record", help="save the HUD view to this .mp4 (off by default: no frames stored)")
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()

    import main as eg
    eg._load()
    ex = eg._extractor
    ex.reset()
    seg = SignSegmenter(fps=eg.CFG.get("target_fps", 15))
    speaker = Speaker(a.tts)

    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        sys.exit("!! could not open the camera / video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = bool(a.video)
    mirror = not a.no_mirror and not is_file
    step = max(1, int(round(src_fps / ex.target_fps))) if is_file else 1
    writer = None

    state = dict(status="LISTENING", caption=None, confidence=None, top=None, latency_ms=None)
    transcript, show_hands = [], True
    result_until, i = 0.0, 0
    t_prev, fps_ema, last_row = time.perf_counter(), None, None
    t_last_ms = time.perf_counter()

    while True:
        ok, bgr = cap.read()
        if not ok:
            if is_file:                                    # end of file: finish the last sign
                sign = seg.flush()
                if sign is not None:
                    probs, ms = eg.classify_frames(sign)
                    k = np.argsort(-probs)[:4]
                    state.update(status="RESULT", caption=eg.VOCAB[k[0]], confidence=float(probs[k[0]]),
                                 top=[(eg.VOCAB[j], float(probs[j])) for j in k], latency_ms=ms)
                    transcript.append(eg.VOCAB[k[0]])
                    print(f"{eg.VOCAB[k[0]]}  ({100 * float(probs[k[0]]):.0f}%, model {ms:.1f} ms)")
            break
        i += 1
        if i % step == 0:
            h, w = bgr.shape[:2]
            k_ = 640 / max(h, w)
            small = cv2.resize(bgr, (int(w * k_), int(h * k_)), interpolation=cv2.INTER_AREA) if k_ < 1 else bgr
            now = time.perf_counter()
            step_ms = max(1, int((now - t_last_ms) * 1000)) if not is_file else int(1000 * step / src_fps)
            t_last_ms = now
            row = ex.frame(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), step_ms)
            last_row = row
            sign = seg.feed(row)
            if sign is not None:
                probs, ms = eg.classify_frames(sign)
                k = np.argsort(-probs)[:4]
                word, conf = eg.VOCAB[k[0]], float(probs[k[0]])
                shown = word if conf >= a.min_conf else "?"
                state.update(status="RESULT", caption=shown, confidence=conf,
                             top=[(eg.VOCAB[j], float(probs[j])) for j in k], latency_ms=ms)
                if shown != "?":
                    transcript.append(word)
                    speaker.say(word)
                result_until = time.perf_counter() + 2.0
            elif seg.active:
                state["status"] = "SIGNING"
            elif time.perf_counter() > result_until:
                state["status"] = "LISTENING"

        view = bgr.copy()
        if show_hands and last_row is not None:
            hud.draw_hands(view, last_row)
        if mirror:
            view = cv2.flip(view, 1)
        if view.shape[1] != a.width:
            view = cv2.resize(view, (a.width, int(view.shape[0] * a.width / view.shape[1])))
        t = time.perf_counter()
        inst = 1.0 / max(t - t_prev, 1e-6)
        t_prev = t
        fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
        hud.draw(view, fps=fps_ema, transcript=transcript, **state)
        if a.record:
            if writer is None:
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps / (1 if not is_file else 1), (view.shape[1], view.shape[0]))
            writer.write(view)
        cv2.imshow("EchoGesture - live captions (q to quit)", view)
        key = cv2.waitKey(1 if not is_file else max(1, int(1000 / src_fps))) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            sign = seg.flush()
            if sign is not None:
                probs, ms = eg.classify_frames(sign)
                k = np.argsort(-probs)[:4]
                state.update(status="RESULT", caption=eg.VOCAB[k[0]], confidence=float(probs[k[0]]),
                             top=[(eg.VOCAB[j], float(probs[j])) for j in k], latency_ms=ms)
                transcript.append(eg.VOCAB[k[0]])
                speaker.say(eg.VOCAB[k[0]])
                result_until = time.perf_counter() + 2.0
        elif key == ord("c"):
            transcript.clear()
        elif key == ord("t"):
            speaker.ok = speaker.available and not speaker.ok
        elif key == ord("l"):
            show_hands = not show_hands
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("transcript:", " ".join(transcript))


if __name__ == "__main__":
    main()
