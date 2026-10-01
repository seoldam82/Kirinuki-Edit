# -*- coding: utf-8 -*-
"""댄스 쇼츠 트래킹 카메라 (Dance(C:S) Solo Shorts, 2026-10-01) - edit.json camera.keys 를 만든다.

사용자: "캐릭터가 화면 밖으로 나갈 것 같은 경우에 ... 풀샷을 잡기 위해서 혹은 캐릭터 중심이 중앙에 오도록, 상하좌우 여백이 많이
생기지 않도록 이동. 필요시 약간의 줌인, 줌아웃을 허용" -> 멈췄다 옮기는 첫 판은 "너무 부자연스러워 / 가능하면 최대한 계속 트래킹
하듯이 캐릭터의 중심을 기준으로 따라가야해". 그래서 카메라는 늘 몸 무게중심을 따라간다.

   -> "최대한 몸 전체가 다 나오게 줌인 아웃" -> "카메라가 캐릭터보다 먼저 이동하거나 늦게 이동하면 안돼".
1. 캐릭터: isnet-anime (~/.u2net/isnet-anime.onnx, 애니 캐릭터 분할) 윤곽을 FPS 마다 (편 폴더 char_masks.npz 캐시),
   그 사이 프레임은 윤곽 점을 광학 흐름으로 앞 · 뒤에서 옮겨 섞는다 -> DFPS 마다 몸 상자 · 무게중심.
   -> "순간적으로 크게 움직이는 이동, 줌인, 아웃이 많으면 어지러우니까 크게 변경되는 경우 오히려 천천히 ... 왔다 갔다 하는 기간 동안
   최대 너비까지 천천히 줌아웃 -> 이후에 다시 일반적으로" + "예시로 준 영상을 참고 해".
2. 카메라 (2026-10-01 셋째, 사용자 정답 두 쌍 n152-0 -> -1, n153-0 -> -2 를 SIFT 로 되찾아 맞댄 것 - 아래 "정답에서 잰 카메라"):
   배율 · 세로 제자리는 keep 구간마다 고정 (몸 세로 가운데 = 화면 가운데, 몸 높이 가운데값이 화면의 FILL0). 몸이 화면을 넘을 때만 넓히고
   (V_ZO) 잠시 쥐었다 천천히 돌아온다. 세로는 손 포함 몸 꼭대기가 위 여백 HEAD_KEEP 를 파고들 때만 따라 올라간다 (큰 점프). 발은 늘 안.
   가로는 무게중심 - 좁은 띠(DZ_C) 안 흔들림은 무시하고 넘으면 떨림만 거른 채 같은 프레임에.
   앞 판들 (참고본 속도 한도 · 계속 트래킹 · 왔다 갔다 구간) 은 "이동, 확대 축소 전부 이상해" 로 버렸다 - PROGRESS.md.
   DFPS 마다 키 (in: linear) - shortsmith 가 이어진 키를 한 경로로 굽는다. 거의 직선인 키는 뺀다.
3. camera.punch (챌린지 순간 확대): [{"s": 12.0, "e": 15.8, "z": 2.0}] - 그 동안 상반신으로 컷 인, 끝나면 컷 아웃.

    python tools/dance_camera.py <편 폴더> [--sheet]      edit.json camera.keys 를 쓰고 요약 · 잘림 검사를 찍는다
                                                        --sheet: camera_sheet.jpg (2초마다 원본에 상자)

수치 근거는 아래 상수 옆에 적는다 (measured = 잰 것, guess = 짐작).
"""
import io, json, os, subprocess, sys
sys.setrecursionlimit(100000)
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

# ---- 수치 ----
# measured = Damyui-n152-1.mov (사용자가 -0.mov 를 세로로 다시 잡은 것) 카메라를 0.1초마다, 몸을 0.25초마다 재서 맞댐
# (알림 그림을 몸으로 잡은 7.9-14.2 · 96.8-97.8초는 뺌)
FPS = 5            # 윤곽(isnet)을 뽑는 간격 - 한 장 1.3초라 프레임마다는 못 돌린다 (CPU 뿐)
DFPS = 30          # 몸 위치 · 카메라 키 간격. 윤곽 사이는 광학 흐름으로 채운다
MW, MH = 480, 270  # 윤곽 · 흐름을 재는 크기
# 사용자 "카메라가 캐릭터보다 먼저 이동하거나 늦게 이동하면 안돼" (2026-10-01): 앞 판은 σ 0.2초 가우스 + 앞뒤 0.6초 창으로
# 카메라가 몸보다 먼저 움직이기 시작하고 늦게 멈췄다. 이제 몸 위치를 프레임마다 재고 같은 프레임에 따라간다
SIG = 0            # 앞뒤를 같이 보는 다듬기는 안 쓴다 - σ 0.05초도 카메라를 한두 프레임 먼저 움직였다. 떨림은 DZ_* 와 가속 한도가 받는다
EDGE_M = 0.01      # 줌아웃은 몸이 이 여백까지 넘을 때만 (화면 밖으로 나가려 할 때)
# 사용자 "아직도 너무 어지러워 순간적으로 크게 움직이는 이동, 줌인, 아웃이 많으면 어지러우니까 크게 변경되는 경우 오히려 천천히 이동 해야 해
# 특히나 왔다 갔다 하는 경우라면 왔다 갔다 하는 기간 동안 최대 너비 까지 천천히 줌아웃 -> 이후에 다시 일반적으로 진행",
# "어떻게 처리하는지 내가 예시로 준 영상을 참고 해". 한도는 -1.mov 카메라 (0.1초마다) 의 99 백분위:
V_X, A_X = 0.31, 1.1   # 가로 속도 h/초 · 가속 h/초². measured 가운데값 0.037 · 90% 0.153 · 99% 0.314 / 가속 99% 1.10
V_Y, A_Y = 0.07, 0.3   # 세로. measured 속도 99% 0.072 (가속 guess)
V_ZO, A_ZO = 0.12, 0.6  # 넓히기 /초. measured 정답 n153 103.2-104.7초 933 -> 1053 (1.5초, 초당 9%), n152 44.4-45.7초 1.3초 램프
TR_FC, TR_BETA, TR_DFC = 1.0, 3.0, 1.0   # 자리 트래킹 거르개 (track): 가만히 1Hz · 속도 h/초마다 +3Hz (guess - 아래 검사 수치로 맞춤)
V_Z, A_Z = 0.035, 0.1  # 당기기 /초. measured 정답 n152 66.2-68.3초 849 -> 910 (초당 3.4%) - 아래 원래 주석은 앞 판 (참고본 99%) 기준.
# 앞 판: 배율 바뀜 /초. measured 99% 0.099 · 최대 0.22 (가속 guess). 다리 차기 64-70초: 2.5초에 걸쳐 7% 넓히고 그대로
DZ_X = 0.025      # 가로: 몸 무게중심이 카메라 가운데에서 너비의 이만큼 안에서 흔들리면 안 따라간다 (넘으면 그 끝을 따라감).
                   # measured -1.mov 카메라-몸 가로 차 사분위 -0.025 ~ 0.015 - 참고본도 이 폭의 흔들림은 안 따라간다
DZ_Y = 0.02        # 세로도 같은 방식 (guess)
TOL = 0.4          # 키를 뺄 때 직선에서 벗어나도 되는 정도 (원본 px)
PUNCH_Z = 2.0      # 챌린지 확대 배율 (guess - 상반신이 차는 정도)
PUNCH_HEAD = 0.14  # 확대 중 머리 위 여백. measured R14ZVrYoWSs 19.2-23.2초 (4.0초, 컷 인 · 컷 아웃): 0.14
HMIN = 0.45        # 상자 높이 최소 (원본 높이 비율) - 너무 당기면 흐리다 (guess)

work = os.path.abspath(sys.argv[1])
E = json.load(io.open(os.path.join(work, "edit.json"), encoding="utf-8"))
SRC = E["source"] if os.path.isabs(E["source"]) else os.path.join(work, E["source"])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
js = ("import {loadPreset} from 'file:///%s/shortsmith/lib/preset.mjs';"
      "const r=loadPreset(process.argv[1], process.argv[2]);process.stdout.write(JSON.stringify({w:r.preset.layout.window,c:r.preset.camera||{}}))") % ROOT.replace(os.sep, "/")
PR = json.loads(subprocess.run(["node", "--input-type=module", "-e", js, E["preset"], work], capture_output=True, text=True, check=True).stdout)
Wn, PCAM = PR["w"], PR["c"]
ASP = Wn["w"] / Wn["h"]
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration",
                     "-of", "csv=p=0", SRC], capture_output=True, text=True).stdout.strip().split(",")
if pr[2] == "N/A":                                 # mkv 는 스트림 길이를 안 적는다 (합본 중간 원본 src_merged.mkv) - 파일 길이로
    pr[2] = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", SRC],
                           capture_output=True, text=True).stdout.strip()
SW, SH, DUR = int(pr[0]), int(pr[1]), float(pr[2])
# 여러 캐릭터가 나란히 춤추는 원본에서 한 명만 따라갈 때: edit.json camera.region = [x0, x1] (원본 가로 비율). 그 띠만 잘라 윤곽을 잡는다 -
# 띠 밖은 몸이 아니다. 가시나0 (2026-10-02): 하늘머리 (왼쪽) 와 검은머리 (오른쪽) 가 둘 다 몸으로 잡혀 상자가 둘을 감쌌다
REG = (E.get("camera") or {}).get("region") or [0, 1]


def seg_region(f, infer, cv2):
    """원본 프레임 한 장 -> region 띠 안에서만 잡은 윤곽 (f 크기)"""
    a, b = int(REG[0] * f.shape[1]), int(REG[1] * f.shape[1])
    out = np.zeros(f.shape[:2], bool)
    out[:, a:b] = seg_frame(f[:, a:b].copy(), infer, cv2)
    return out


def keep_ranges():
    K = E.get("keep") or [[0, DUR]]
    out = []
    for k in K:
        if isinstance(k, dict):
            if k.get("gap") or k.get("source"):
                continue
            out.append((k["s"], k["e"]))
        else:
            out.append((k[0], k[1]))
    return out


# ---- 1. 캐릭터: 0.2초마다 윤곽 (isnet), 그 사이는 광학 흐름으로 프레임마다 ----
def seg_frame(f, infer, cv2):
    """원본 프레임 (RGB) 한 장 -> 몸 윤곽 (MW x MH bool). f 가 원본 일부 (crop) 면 그 크기 그대로 돌려준다"""
    H, W = f.shape[:2]
    im = cv2.resize(f, (1024, 1024)).astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)
    o = infer(im.transpose(2, 0, 1)[None])
    if o.max() < 0.5:
        # 모델이 캐릭터를 못 찾은 장 - 펴서 (정규화) 쓰면 잡음이 윤곽이 된다. 사준완260921 5.6초: 출력 최대 0.099 -> 109점짜리
        # "몸" 을 잡아 카메라가 0.3초 확 당겼다 놓았다 (같은 장을 따로 넣으면 정상 - 이어 읽은 그 프레임만). 빈 윤곽으로 두면
        # dense() 가 앞뒤 윤곽 사이를 흐름으로 잇는다
        return np.zeros((H, W), bool)
    o = (o - o.min()) / (o.max() - o.min() + 1e-9)
    m = (cv2.resize(o, (W, H)) > 0.5).astype(np.uint8)
    n, lab, s_, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros((H, W), bool)
    if n > 1:
        k = 1 + int(np.argmax(s_[1:, cv2.CC_STAT_AREA]))
        bx0, bx1 = s_[k, 0], s_[k, 0] + s_[k, 2]
        # 떨어진 손 · 머리칼은 넣고, 몸 밖의 큰 덩이는 뺀다 - 구독 · 후원 알림의 치비 그림을 몸으로 잡았다
        # (Damyui-n152 7.9-14.2초: 몸 왼쪽 0.3 너비 옆, 몸 넓이의 20% 넘음)
        by0, by1, bh = s_[k, 1], s_[k, 1] + s_[k, 3], s_[k, 3]
        def near(j, pad):
            return (s_[j, 0] < bx1 + pad and s_[j, 0] + s_[j, 2] > bx0 - pad and
                    s_[j, 1] < by1 + pad and s_[j, 1] + s_[j, 3] > by0 - pad)
        A = s_[k, cv2.CC_STAT_AREA]
        keep = [k] + [j for j in range(1, n) if j != k and s_[j, cv2.CC_STAT_AREA] > 0.01 * A and
                      (near(j, 0) if s_[j, cv2.CC_STAT_AREA] > 0.1 * A else near(j, 0.15 * bh))]
        out = np.isin(lab, keep)
    return out


def body_box(m):
    """윤곽 -> (x0, y0, x1, y1, 넓이) MW x MH 칸, 빈 윤곽은 None"""
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, len(xs))


def same_body(p, q, gap):
    """q 가 p 와 같은 몸인가 - 자리가 갑자기 옮겨 가거나 (구독 알림 그림만 잡음) 넓이가 확 커지면 (알림 + 몸) 아니다.
    measured 260921 합본 45-51초 (터미널): 알림 치비 그림만 잡은 장은 가운데가 몸 높이의 0.8 옆, 몸과 붙여 잡은 장은 넓이 1.67배.
    measured 댄스 편 6개 0.2초 사이: 가운데 이동 (가로 + 세로) 99% 0.11-0.26 · 최대 0.37 몸 높이, 넓이 99% 0.82-1.23배
    (한 편만 0.50-2.02) -> 한도 0.45 + 0.2초마다 0.1 · 0.6-1.5배 (guess)"""
    h = p[3] - p[1]
    d = abs((p[0] + p[2]) / 2 - (q[0] + q[2]) / 2) + abs((p[1] + p[3]) / 2 - (q[1] + q[3]) / 2)
    return d < h * (0.45 + 0.5 * gap) and 0.6 < q[4] / p[4] < 1.5


def check_masks(have, want, infer, W, H):
    """keep 구간마다 시간 순으로 윤곽이 앞 몸과 이어지는지 본다. 끊긴 장은 앞 몸 둘레만 잘라 (가로 몸 높이 1.5배) 다시 잡고,
    그래도 아니면 빈 윤곽으로 둔다 (dense() 가 흐름으로 잇는다). 구간 첫 장은 그대로 믿는다"""
    import cv2
    fixed = bad = 0
    R = keep_ranges()
    for ri, (s, e) in enumerate(R):
        # 이어 붙은 keep (여러 원본을 이은 합본) 은 경계에서 다른 장면이 된다 - 그 너머 장은 다음 구간이 처음부터 본다
        e2 = R[ri + 1][0] if ri + 1 < len(R) and R[ri + 1][0] <= e + 0.4 else e + 0.4
        prev, pt = None, None
        for t in [t for t in want if s - 1e-6 <= t < e2 - 1e-6 and t in have]:
            q = body_box(have[t])
            if q is None:
                continue
            if prev is None or same_body(prev, q, t - pt):
                prev, pt = q, t
                continue
            h = prev[3] - prev[1]
            cx = (prev[0] + prev[2]) / 2
            a, b = int(max(REG[0] * MW, cx - 0.75 * h) * W / MW), int(min(REG[1] * MW, cx + 0.75 * h) * W / MW)
            fr = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.3f" % t, "-i", SRC, "-frames:v", "1", "-vf", "scale=%d:%d" % (W, H),
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            full = np.zeros((H, W), bool)
            if len(fr) == W * H * 3 and b - a > 16:
                full[:, a:b] = seg_frame(np.frombuffer(fr, np.uint8).reshape(H, W, 3)[:, a:b].copy(), infer, cv2)
            m = cv2.resize(full.astype(np.uint8), (MW, MH), interpolation=cv2.INTER_NEAREST).astype(bool)
            q = body_box(m)
            if q is not None and same_body(prev, q, t - pt):
                have[t] = m; prev, pt = q, t; fixed += 1
            else:
                have[t] = np.zeros((MH, MW), bool); bad += 1
    if fixed or bad:
        print("  윤곽: 몸이 아닌 것을 잡은 장 %d - 몸 둘레만 다시 잡음 %d, 비움 %d" % (fixed + bad, fixed, bad), flush=True)


def isnet_runner():
    """isnet-anime 한 장 -> 1024x1024 출력. 외장 GPU (OpenVINO) 가 있으면 거기서, 없으면 onnxruntime CPU.
    measured i7-1260P / Arc A350M: CPU 0.97초/장 (스레드 · 세션 수를 바꿔도 0.90-0.97 - 이미 꽉 참), Arc 0.070, 내장 Iris Xe 0.23.
    GPU 는 반정밀도인데 윤곽 IoU 0.9994-1.0000 (허니하트 6장, CPU 와 견줌). 컴파일 결과는 ~/.cache/openvino 에 (첫 판 ~10초, 그 뒤 ~4초)"""
    model = os.path.expanduser("~/.u2net/isnet-anime.onnx")
    try:
        import openvino as ov
        core = ov.Core()
        dev = next(d for d in core.available_devices if d.startswith("GPU") and "dGPU" in core.get_property(d, "FULL_DEVICE_NAME"))
        core.set_property({"CACHE_DIR": os.path.expanduser("~/.cache/openvino")})
        cm = core.compile_model(model, dev)
        print("  윤곽: OpenVINO %s (%s)" % (dev, core.get_property(dev, "FULL_DEVICE_NAME")), flush=True)
        return lambda x: cm(x)[0][0, 0]
    except Exception as e:                         # openvino 없음 · 외장 GPU 없음 (StopIteration) · 드라이버 문제
        import onnxruntime as ort
        print("  윤곽: onnxruntime CPU (%s)" % (type(e).__name__,), flush=True)
        sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        return lambda x: sess.run(None, {"img": x})[0][0, 0]


def masks_isnet():
    """FPS 마다 몸 윤곽 (MW x MH bool). 편 폴더 char_masks.npz 에 캐시"""
    st = os.stat(SRC)
    key = json.dumps({"src": SRC.replace(os.sep, "/"), "size": st.st_size, "mtime": int(st.st_mtime), "fps": FPS, "w": MW, "v": 7, **({"region": REG} if REG != [0, 1] else {})})
    cache = os.path.join(work, "char_masks.npz")
    have = {}
    if os.path.exists(cache):
        z = np.load(cache)
        if str(z["key"]) == key:
            have = {round(float(t), 3): np.unpackbits(b)[:MW * MH].reshape(MH, MW).astype(bool) for t, b in zip(z["t"], z["m"])}
    want = sorted({round(round(t * FPS) / FPS, 3) for s, e in keep_ranges()
                   for t in np.arange(max(0, s - 0.4), min(DUR, e + 0.4), 1 / FPS)})
    todo = [t for t in want if t not in have]
    if todo:
        import cv2
        infer = isnet_runner()
        W, H = 1024, int(round(1024 * SH / SW))
        groups, g = [], [todo[0]]                  # 이어진 덩이마다 ffmpeg 한 번
        for t in todo[1:]:
            if t - g[-1] > 1.5 / FPS:
                groups.append(g); g = [t]
            else:
                g.append(t)
        groups.append(g)
        done = 0
        seg = lambda f: cv2.resize(seg_region(f, infer, cv2).astype(np.uint8), (MW, MH), interpolation=cv2.INTER_NEAREST).astype(bool)
        for g in groups:
            p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", "%.3f" % g[0], "-i", SRC, "-t", "%.3f" % (g[-1] - g[0] + 0.5 / FPS),
                                  # round=up: fps 필터는 칸 안의 마지막 프레임을 내놓는다 - 기본 (near) 으로는 t 라고 적은 윤곽이
                                  # 실제로 t + 0.083초 (60fps 원본) 장이었다. 윤곽 시각은 미래, 그 사이 흐름은 제 시각이라 몸 자리가
                                  # 0.2초마다 앞섰다 돌아왔다 - 카메라가 몸보다 먼저 · 늦게 갔다 (measured 260921 합본 21.2-21.6초, IoU 최대 자리)
                                  "-vf", "fps=%d:round=up,scale=%d:%d" % (FPS, W, H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
            for t in g:
                b = p.stdout.read(W * H * 3)
                if len(b) < W * H * 3:
                    break
                have[t] = seg(np.frombuffer(b, np.uint8).reshape(H, W, 3))
                done += 1
                if done % 50 == 0:
                    print("  윤곽 %d/%d" % (done, len(todo)), flush=True)
            p.kill()
        check_masks(have, want, infer, W, H)
        ts = sorted(have)
        np.savez_compressed(cache, key=key, t=np.array(ts), m=np.array([np.packbits(have[t].ravel()) for t in ts]))
    return {t: have[t] for t in want if t in have}


def stats(pts):
    """점들 -> [x0, y0, x1, y1, 무게중심 x, 머리 꼭대기 y, 상반신 무게중심 x] (원본 px). 끝은 0.5 / 99.5 백분위 - 흐름이 튄 점
    몇 개에 안 끌려가게. 머리 꼭대기 = 무게중심 둘레 띠 (몸 높이의 0.08 양옆) 의 윤곽 꼭대기 - 팔을 들면 손이 몸 상자 꼭대기가 된다.
    상반신 = 머리 꼭대기부터 몸 높이의 0.35 까지. 뒤 두 칸은 C:D 클로즈업 (dance_beats.py) 이 쓴다"""
    if len(pts) < 20:
        return [np.nan] * 7
    # 윤곽은 원본 비율과 상관없이 MW x MH 로 늘여 둔다 - 가로 세로 배율이 다르다. 한 배율(SW/MW)로 쓰면 16:9 가 아닌 원본에서
    # 몸이 세로로 줄어 발이 실제보다 높게 잡혔다 (260921 합본 1726x1080: 발 105px 위, 완성본 90% 프레임에서 발목 아래가 잘렸다)
    sx, sy = SW / MW, SH / MH
    x0, x1 = np.percentile(pts[:, 0], [0.5, 99.5]); y0, y1 = np.percentile(pts[:, 1], [0.5, 99.5])
    cx, bh = pts[:, 0].mean(), y1 - y0
    band = pts[np.abs(pts[:, 0] - cx) < 0.08 * bh]
    hy = np.percentile(band[:, 1], 0.5) if len(band) >= 10 else y0
    up = pts[pts[:, 1] < hy + 0.35 * bh]
    ux = up[:, 0].mean() if len(up) >= 10 else cx
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy, cx * sx, hy * sy, ux * sx]


from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(8)       # cv2 는 GIL 을 놓는다 - measured Farneback 480x270 19.8 -> 5.0ms/장 (i7-1260P)


def dense():
    """DFPS 마다 몸 [x0, y0, x1, y1, cx]. 윤곽이 있는 시각은 윤곽 그대로, 그 사이는 앞 윤곽의 점을 흐름대로 앞으로 옮긴 것과
    뒤 윤곽의 점을 거꾸로 옮긴 것을 시간 비율로 섞는다 (한쪽만 쓰면 0.2초 사이에 흐름 오차가 쌓인다)"""
    import cv2
    M = masks_isnet()
    mt = np.array(sorted(M))
    T, V = [], []
    R = keep_ranges()
    for ri, (s, e) in enumerate(R):
        a0 = max(0.0, round(round((s - 0.2) * FPS) / FPS, 3))
        if ri and R[ri - 1][1] >= s - 1e-6:
            a0 = s                                 # 앞 구간과 이어 붙었다 - 경계 앞은 다른 장면이라 윤곽 · 흐름을 넘겨 오지 않는다
        dur = min(DUR, e + 0.4) - a0
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.3f" % a0, "-i", SRC, "-t", "%.3f" % dur, "-vf",
                              "fps=%d,scale=%d:%d,format=gray" % (DFPS, MW, MH), "-f", "rawvideo", "-"], capture_output=True).stdout
        F = np.frombuffer(raw, np.uint8).reshape(-1, MH, MW)
        ft = a0 + np.arange(len(F)) / DFPS
        flow = lambda ij: cv2.calcOpticalFlowFarneback(F[ij[0]], F[ij[1]], None, 0.5, 3, 21, 3, 5, 1.1, 0)
        anchors = [(int(round((t - a0) * DFPS)), t) for t in mt if a0 - 1e-6 <= t <= ft[-1] + 1e-6 and M[t].any()
                   and int(round((t - a0) * DFPS)) < len(F)]   # 빈 윤곽은 건너뛴다
        vals = np.full((len(F), 7), np.nan)
        for ia, ta in anchors:
            vals[ia] = stats(np.argwhere(M[ta])[:, ::-1].astype(np.float32))
        for (ia, ta), (ib, tb) in zip(anchors, anchors[1:]):
            if ib - ia < 2:
                continue
            fw = {}
            # 이 구간에 쓸 흐름 (앞으로 · 뒤로) 을 스레드 8개로 먼저 - 한 줄로 돌리면 23초 (허니하트 1160번), 나눠 돌려도 값은 같다
            need = [(i, i + 1) for i in range(ia, ib - 1)] + [(i, i - 1) for i in range(ib, ia + 1, -1)]
            FL = dict(zip(need, POOL.map(flow, need)))
            p = np.argwhere(M[ta])[::3, ::-1].astype(np.float32)
            for i in range(ia, ib - 1):
                fl = FL[(i, i + 1)]
                q = np.clip(p.round().astype(int), 0, [MW - 1, MH - 1])
                p = p + fl[q[:, 1], q[:, 0]]
                fw[i + 1] = stats(p)
            p = np.argwhere(M[tb])[::3, ::-1].astype(np.float32)
            for i in range(ib, ia + 1, -1):
                fl = FL[(i, i - 1)]
                q = np.clip(p.round().astype(int), 0, [MW - 1, MH - 1])
                p = p + fl[q[:, 1], q[:, 0]]
                u = (i - 1 - ia) / (ib - ia)
                vals[i - 1] = (1 - u) * np.array(fw[i - 1]) + u * np.array(stats(p))
        ix = np.arange(len(F))                     # 첫 윤곽 앞 · 끝 윤곽 뒤 · 못 찾은 칸은 이 구간 안의 이웃으로
        for c in range(vals.shape[1]):
            ok = ~np.isnan(vals[:, c])
            if ok.any():
                vals[:, c] = np.interp(ix, ix[ok], vals[ok, c])
        # 다음 keep 이 바로 이어 붙으면 (원본을 이어 만든 합본) 경계 프레임은 다음 구간 것이다 - 두 번 넣으면 같은 시각 키가 둘
        nxt = R[ri + 1][0] if ri + 1 < len(R) and R[ri + 1][0] <= e + 1e-6 else None
        keep = (ft >= s - 1e-6) & ((ft < nxt - 1e-6) if nxt is not None else (ft <= e + 1e-6))
        T.extend(ft[keep]); V.extend(vals[keep])
    T, V = np.array(T), np.array(V)
    for c in range(V.shape[1]):                    # 못 찾은 칸은 이웃으로
        ok = ~np.isnan(V[:, c])
        V[:, c] = np.interp(T, T[ok], V[ok, c])
    return T, V


T, B = dense()
x0, y0, x1, y1, mcx = B[:, :5].T

# ---- 2. 카메라 ----
# 작은 움직임은 몸과 같은 프레임에, 큰 움직임은 참고본의 속도 · 가속 한도 안에서 천천히 (넘치는 만큼만 늦는다 - 미리 움직이지는 않는다).
# 왔다 갔다 하는 구간은 그 동안 몸이 오가는 전체 폭이 들어가게 천천히 넓혀 두고, 끝나면 다시 평소대로.
cam = E.get("camera") or {}
punch = cam.get("punch") or []
sg = lambda a, sec: gaussian_filter1d(a, sec * DFPS) if sec > 0 else a
# keep 구간이 둘 넘으면 구간마다 따로 찍은 장면이다 (여러 원본을 이어 만든 합본 - 260921 사준완 · 제로투 · 터미널, 2026-10-01).
# 구간 첫 프레임에서 카메라 상태 (따라가기 · 흔들림 무시 · 배율) 를 새로 시작하고 거기서 컷 - 이어 따라가면 앞 장면 자리에서 미끄러져 온다
KSTART = {int(np.searchsorted(T, s0 - 1e-6)): s0 for s0, _ in keep_ranges()[1:]}
KSTART = {i: s0 for i, s0 in KSTART.items() if 0 < i < len(T)}
RESET = set(KSTART)
SEGSTART = np.maximum.accumulate(np.where(np.isin(np.arange(len(T)), list(RESET | {0})), np.arange(len(T)), 0))
X0, Y0, X1, Y1, MC = (sg(v, SIG) for v in (x0, y0, x1, y1, mcx))


# ---- 사용자 정답 두 쌍에서 잰 카메라 (2026-10-01 셋째) ----
# 사용자 "이동, 확대 축소 전부 이상해 ... 답 있으니까 다시 분석해 왜 이렇게 여백 중심 이런 것도 제대로 안 맞추는 거야?"
# 정답 = 사용자가 손으로 다시 잡은 Damyui-n152-1.mov (원본 -0) · Damyui-n153-2.mov (원본 -0, 0-120초). 프레임마다 SIFT 로 원본 속
# 상자를 되찾아 (10fps) 몸 (이 도구의 윤곽) 과 맞댔다. 잰 것:
#  - 배율은 거의 고정: 같은 배율로 있는 시간 n152 91% · n153 87% (우리 앞 판 77-82%, 초당 0.34번 바뀜). 바꿀 때는 1-2초 직선 램프
#  - 세로도 거의 고정: 멈춰 있는 시간 86% · 84% (앞 판 46% - 몸 따라 오르내렸다). 몸 세로 가운데가 화면 가운데 (0.50-0.53)
#  - 큰 점프만 따라 올라간다 (n153 59.5-60.2초: 머리 여백이 0.11 아래로 줄 때만 머리를 따라 올라가 그 여백을 지키고, 내려오면 제자리)
#  - 가로는 무게중심, 늦음 0 (10fps 상호상관), 어긋남 가운데값 높이의 0.01. 멈춰 있는 시간 39% · 46%
#  - 몸 (머리 꼭대기-발) / 화면 높이 가운데값 0.78-0.87 (구간마다) - 0.86 으로 두면 배율이 정답과 1.00 · 1.04
FILL0 = 0.82       # 구간 배율: 몸 높이 가운데값이 화면의 이만큼. 0.86 (정답 배율 1.00 · 1.04) 은 위아래 여백이 0.07 · 0.06 뿐이라 조금 앉거나
                   # 뛰어도 세로가 움직였다 ("뚝뚝"). 0.82: 정답과 배율 1.05 · 1.04, 세로 멈춤 87% · 85% (정답 86 · 84), 합본 꺾임 51 -> 5 (2026-10-01 넷째)
FILL_HI = 0.97     # 그리고 몸 높이 98 백분위가 이만큼 안 (guess - 정답과 배율 1.00 · 1.04, 잘림 1-2%)
WIDE_HOLD = 2.0    # 몸이 화면을 넘어 넓힌 배율은 이만큼 쥐고 있다가 천천히 돌아온다 (guess - n153 104-106초 넓힌 뒤 그대로 둠)
HEAD_KEEP = 0.045  # 몸 꼭대기 (위로 든 손 포함) 위 여백이 이만큼 아래로 줄 때만 따라 올라간다. measured 정답 손 포함 위 여백 1 백분위 0.041 · 0.042. 정답은 평소 춤의 끄덕임에는 안 움직였다 (n152 머리 여백 5 백분위 0.001
                   # 인데 그대로) - 평소 여백의 0.6 으로 두니 세로가 33% 시간 움직였다 (정답 14%). 큰 점프는 n153 59.5초처럼 머리를 따라 올라간다
FOOT_KEEP = 0.02   # 발 아래 최소 여백
KNEE = 0.04        # 세로 무른 무릎 (높이 비율): 여백 선에 이만큼 다가오면서부터 천천히 붙기 시작하고 지나면 같이 간다 - 딱 닿는 순간
                   # 멈춤 -> 몸 속도로 바뀌어 "점프하거나 조금이라도 앉으면 ... 뚝뚝" (2026-10-01 넷째, 합본 20.3초 0.15초에 35px) (guess)
V_FC, V_BETA = 1.0, 1.5   # 세로 떨림 거르개 (one-euro, Hz · h/초당 Hz) - 무릎이 남긴 꺾임을 펴고, 늦은 만큼은 아래 잘림 선이 받는다.
                          # 2.0 · 4.0 은 합본 꺾임 6 · 방향 바뀜 초당 0.13, 1.0 · 1.5 는 5 · 0.09 (머리 · 발 잘림 0)
CUT_M = 0.005      # 거르개를 거친 뒤에도 이 여백 아래로는 안 간다 (머리 · 발 잘림 선)
DZ_C = 0.01        # 가로 흔들림 무시 띠 (높이 비율) - 정답처럼 멈춰 있는 시간 (39-46%) 이 생기게 (앞 판 32%)


def dead(target, band, reset=()):
    """band 안의 흔들림은 무시 - 밖으로 나가면 그 끝을 따라간다 (지난 값만 쓴다). reset 프레임에서는 그 자리에서 새로"""
    out, c = np.empty(len(target)), float(target[0])
    for i, tg in enumerate(target):
        if i in reset:
            c = float(tg)
        c = min(max(c, tg - band[i]), tg + band[i])
        out[i] = c
    return out


def follow(target, vmax, amax, scale, reset=()):
    vmax, amax = np.broadcast_to(vmax, target.shape), np.broadcast_to(amax, target.shape)
    """target 을 따라간다. 속도 vmax · 가속 amax (scale 곱) 안이면 같은 프레임에 딱 맞고, 넘치면 그만큼만 천천히.
    멈출 거리를 봐서 넘어가지 않게 줄인다. 앞 프레임 값만 쓴다 (미리 움직이지 않는다)"""
    dt = 1.0 / DFPS
    c, v, out = float(target[0]), 0.0, np.empty(len(target))
    for i, tg in enumerate(target):
        if i in reset:                              # 컷: 그 자리에서 멈춘 채로 새로
            c, v = float(tg), 0.0
        vm, am = vmax[i] * scale[i], amax[i] * scale[i]
        e = tg - c
        vd = np.sign(e) * min(vm, np.sqrt(2 * am * abs(e)), abs(e) / dt)
        v += np.clip(vd - v, -am * dt, am * dt)
        c2 = c + v * dt
        if (c2 - tg) * (c - tg) < 0:                # 목표를 지나치지 않는다 - 줌이 몸보다 2px 넓어진 적이 있다
            c2, v = float(tg), 0.0
        c = c2
        out[i] = c
    return out


def zoom_follow(target, reset=()):
    """배율: 넓히기 (몸이 화면을 넘음) 는 V_ZO 로 빨리 - 발 · 머리를 자르지 않는 게 먼저다. 당기기는 참고본 빠르기 V_Z 로 천천히"""
    dt = 1.0 / DFPS
    c, v, out = float(target[0]), 0.0, np.empty(len(target))
    for i, tg in enumerate(target):
        if i in reset:
            c, v = float(tg), 0.0
        vm, am = (V_ZO, A_ZO) if tg > c else (V_Z, A_Z)
        vm, am = vm * c, am * c
        e = tg - c
        vd = np.sign(e) * min(vm, np.sqrt(2 * am * abs(e)), abs(e) / dt)
        v += np.clip(vd - v, -am * dt, am * dt)
        c2 = c + v * dt
        if (c2 - tg) * (c - tg) < 0:
            c2, v = float(tg), 0.0
        c = c2
        out[i] = c
    return out


def track(target, scale, reset=(), fc=None, beta=None):
    """자리 트래킹 (one-euro 거르개, 지난 값만): 가만히 있을 때는 TR_FC Hz 로 눌러 떨림을 없애고, 빨리 움직일수록 거르는 띠를
    넓혀 (TR_BETA x 속도 h/초) 몸과 같은 프레임에 붙는다. 미리 움직이지 않는다 - 지난 프레임만 본다"""
    dt = 1.0 / DFPS
    a_of = lambda fc: 1.0 / (1.0 + 1.0 / (2 * np.pi * fc * dt))
    c, d, out = float(target[0]), 0.0, np.empty(len(target))
    for i, tg in enumerate(target):
        if i in reset:
            c, d = float(tg), 0.0
        d += a_of(TR_DFC) * ((tg - c) / dt - d)
        c += a_of((TR_FC if fc is None else fc) + (TR_BETA if beta is None else beta) * abs(d) / scale[i]) * (tg - c)
        out[i] = c
    return out


# 구간 (keep) 마다: 배율 · 세로 제자리를 구간 전체 몸으로 한 번 정한다 (고정 - 움직이는 게 아니라 미리 움직이는 일이 없다)
full = y1 - B[:, 5]                                # 머리 꼭대기 - 발 (위로 든 손 빼고)
segs = sorted(RESET | {0}) + [len(T)]
H0, C0 = np.empty(len(T)), np.empty(len(T))
for a, b in zip(segs, segs[1:]):
    f = full[a:b]
    h0 = np.clip(max(np.median(f) / FILL0, np.percentile(f, 98) / FILL_HI, np.percentile(x1[a:b] - x0[a:b], 90) / (ASP * 0.96)), HMIN * SH, SH)
    c0 = np.median((B[a:b, 5] + y1[a:b]) / 2)        # 몸 세로 가운데를 화면 가운데에
    H0[a:b], C0[a:b] = h0, c0
# 배율: 평소는 H0. 몸이 화면을 넘으면 (머리-발 또는 너비) 빨리 넓히고, WIDE_HOLD 동안 쥐고 있다가 천천히 돌아온다
need = np.maximum(full / (1 - 2 * FOOT_KEEP), (x1 - x0) / (ASP * (1 - 2 * EDGE_M)))
hold = np.array([need[max(int(SEGSTART[i]), i - int(WIDE_HOLD * DFPS)):i + 1].max() for i in range(len(T))])
H = np.clip(zoom_follow(np.maximum(H0, hold), RESET), HMIN * SH, SH)
Wd = H * ASP
# 가로: 무게중심을 흔들림 무시 띠 + 떨림 거르개로 (지난 값만 - 먼저 안 간다), 몸 끝이 나가면 그 프레임에 밀어 넣는다
CX = track(dead(mcx, DZ_C * H, RESET), H, RESET)
lo, hi = x1 - Wd / 2 + EDGE_M * Wd, x0 + Wd / 2 - EDGE_M * Wd
CX = np.clip(np.where(lo <= hi, np.clip(CX, lo, hi), (lo + hi) / 2), Wd / 2, SW - Wd / 2)
# 세로: 제자리 (몸 가운데 = 화면 가운데). 머리가 위 여백을 파고들 때만 머리를 따라 올라가고 내려오면 제자리, 발은 늘 안에
def knee(p, w):
    """p > 0 만큼 비켜야 할 때 비킬 양. -w 부터 천천히 (속도가 이어지게 2차 곡선), w 넘으면 p 그대로. 늘 p 이상이라 여백은 지킨다"""
    return np.where(p <= -w, 0.0, np.where(p >= w, p, (p + w) ** 2 / (4 * w)))


home = C0 - H / 2
TOP = home - knee(home - (y0 - HEAD_KEEP * H), KNEE * H)          # 위: 손 포함 꼭대기가 여백을 파고들면 올라간다
TOP = TOP + knee((y1 + FOOT_KEEP * H - H) - TOP, KNEE * H)         # 아래: 발이 여백을 파고들면 내려간다 (발이 먼저)
TOP = track(TOP, H, RESET, V_FC, V_BETA)
TOP = np.minimum(TOP, B[:, 5] - CUT_M * H)                        # 잘림 선은 머리 - 번쩍 뛰며 든 손끝은 한순간 잘려도 꺾지 않는다 (21.4초)
TOP = np.maximum(TOP, y1 + CUT_M * H - H)
TOP = np.clip(TOP, 0, SH - H)
X = CX - Wd / 2
BASE = float(np.median(H0))

# C:D (프리셋 camera.mode "dynamic"): 위 트래킹을 베이스로 비트에 맞춰 컷 인 · 밀기 · 빼기 · 순간 확대 (tools/dance_beats.py)
CUT, DYN = np.zeros(len(T), bool), None
if PCAM.get("mode") == "dynamic":
    import dance_beats
    X, TOP, H, CUT, DYN = dance_beats.plan(T, B, (X, TOP, H), dict(
        SW=SW, SH=SH, ASP=ASP, DFPS=DFPS, follow=follow, dead=dead, src=SRC, work=work,
        DZ_X=DZ_X, DZ_Y=DZ_Y, V_X=V_X, A_X=A_X, V_Y=V_Y, A_Y=A_Y))
CUT[sorted(RESET)] = True                          # keep 구간 시작은 늘 컷

# 거의 직선인 키는 뺀다 (Douglas-Peucker, x · y · h 가 TOL 안)
def simplify(i, j, out):
    if j <= i + 1:
        return
    u = (T[i + 1:j] - T[i]) / (T[j] - T[i])
    err = np.max(np.abs(np.stack([X[i + 1:j] - (X[i] + (X[j] - X[i]) * u), TOP[i + 1:j] - (TOP[i] + (TOP[j] - TOP[i]) * u),
                                  H[i + 1:j] - (H[i] + (H[j] - H[i]) * u)])), axis=0)
    k = i + 1 + int(np.argmax(err))
    if err[k - i - 1] > TOL:
        simplify(i, k, out); out.append(k); simplify(k, j, out)

box = lambda i: {"x": round(float(X[i]), 1), "y": round(float(TOP[i]), 1), "h": round(float(H[i]), 1)}
keys = []
segs = [0] + [i for i in range(1, len(T)) if CUT[i]] + [len(T)]
for a, b in zip(segs, segs[1:]):                   # 컷마다 끊는다 - 조각 첫 키는 in 없이 (그 프레임에 컷)
    idx = [a]
    simplify(a, b - 1, idx)
    idx = sorted(set(idx + [b - 1]))
    t0 = KSTART.get(a, float(T[idx[0]]))          # keep 구간 첫 키는 구간 시작에 딱 (키 격자 1/30초라 한 프레임 늦게 컷이 났다)
    keys += [dict(t=round(t0, 3), **box(idx[0]))] + [dict(t=round(float(T[i]), 3), **box(i), **{"in": "linear"}) for i in idx[1:]]

# 챌린지 순간 확대: 그 동안 상반신으로 컷 인 (고정), 끝나면 그 시각 트래킹 자리로 컷 아웃
for p in punch:
    s, e, z = p["s"], p["e"], p.get("z", PUNCH_Z)
    m = (T >= s) & (T < e)
    i1 = min(len(T) - 1, int(np.searchsorted(T, e)))
    h = max(HMIN * SH, H[i1] / z)
    w = h * ASP
    pc = {"x": round(float(np.clip(np.median(mcx[m]) - w / 2, 0, SW - w)), 1),
          "y": round(float(np.clip(np.min(y0[m]) - PUNCH_HEAD * h, 0, SH - h)), 1), "h": round(float(h), 1)}
    keys = [k for k in keys if not (s - 1e-6 <= k["t"] <= e + 1e-6)]
    keys.append(dict(t=round(s, 3), **pc))
    keys.append(dict(t=round(e, 3), **box(i1)))
    keys.sort(key=lambda k: k["t"])
    # 확대 바로 뒤 키가 확대 끝에서 이어지도록 (in 그대로), 확대 바로 앞은 그 자리에서 끊긴다 (컷)

# ---- 3. 검사 · 쓰기 ----
def cam_at(t):
    k0 = [k for k in keys if k["t"] <= t + 1e-6]
    a = k0[-1] if k0 else keys[0]
    nx = [k for k in keys if k["t"] > t + 1e-6]
    if nx and nx[0].get("in") == "linear":
        b = nx[0]; u = (t - a["t"]) / (b["t"] - a["t"])
        return {c: a[c] + (b[c] - a[c]) * u for c in ("x", "y", "h")}
    return a


clip = []
for t, b in zip(T, B):
    if not any(s <= t <= e for s, e in keep_ranges()) or any(p["s"] <= t < p["e"] for p in punch) or             (DYN is not None and DYN["kind"][int(np.argmin(abs(T - t)))] != "base"):
        continue                                   # 확대 · 클로즈업 중에는 몸이 잘리는 게 맞다
    k = cam_at(t)
    w = k["h"] * ASP
    over = max(k["x"] - b[0], b[2] - k["x"] - w, k["y"] - b[1], b[3] - k["y"] - k["h"])
    if over > 0.01 * k["h"]:
        clip.append((t, over / k["h"]))

cam["keys"] = keys
cam["basis"] = "tools/dance_camera.py (isnet-anime %dfps + 광학 흐름 %dfps, 구간 고정 배율 FILL0 %.2f, 위 여백 %.3f)" % (FPS, DFPS, FILL0, HEAD_KEEP)
E["camera"] = cam
io.open(os.path.join(work, "edit.json"), "w", encoding="utf-8").write(json.dumps(E, ensure_ascii=False, indent=2) + "\n")
hs = [k["h"] for k in keys]
tot = sum(e - s for s, e in keep_ranges())
sp = np.abs(np.diff(CX)) * DFPS / H[1:]
print("카메라: 트래킹 키 %d (%.1f초) · 확대 %d · 배율 h %d-%d (가운데값 %d, 원본 %d) · 가로 속도 가운데값 %.3f / 95%% %.3f h/초"
      % (len(keys), tot, len(punch), min(hs), max(hs), BASE, SH, np.median(sp), np.percentile(sp, 95)))
if DYN is not None:
    print("비트: %.1f BPM · 박 %d · 마디 %d · 센스 %.0f%% (덩이 %d) · 순간 확대 %d번 · 컷 %d" % (DYN["bpm"], DYN["beats"], DYN["bars"],
          DYN["sense"] * 100, DYN["runs"], len(DYN["pulses"]), int(CUT.sum())))
    print("  손이 움직이는 시간 %.0f%%" % (DYN["hand_dyn"] * 100))
    print("  샷 시간: " + " · ".join("%s %.0f%%" % (k, v * 100) for k, v in DYN["shots"].items()))
print("몸이 잘리는 순간 (1%% 넘게): %d%s" % (len(clip), "" if not clip else "  " + ", ".join("%.1f초 %.0f%%" % (t, o * 100) for t, o in clip[:12])))

if "--sheet" in sys.argv:
    import cv2
    shots = []
    for t in np.arange(keep_ranges()[0][0], keep_ranges()[-1][1], 2.0):
        fr = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.2f" % t, "-i", SRC, "-frames:v", "1", "-vf", "scale=640:-2",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
        h6 = len(fr) // (640 * 3)
        im = np.frombuffer(fr, np.uint8).reshape(h6, 640, 3).copy()
        k, sc = cam_at(t), 640 / SW
        cv2.rectangle(im, (int(k["x"] * sc), int(k["y"] * sc)), (int((k["x"] + k["h"] * ASP) * sc), int((k["y"] + k["h"]) * sc)), (0, 255, 255), 2)
        i = int(np.argmin(abs(T - t)))
        cv2.rectangle(im, (int(B[i, 0] * sc), int(B[i, 1] * sc)), (int(B[i, 2] * sc), int(B[i, 3] * sc)), (255, 0, 255), 1)
        cv2.putText(im, "%.0f" % t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        shots.append(im)
    while len(shots) % 4:
        shots.append(np.zeros_like(shots[0]))
    sheet = np.vstack([np.hstack(shots[r:r + 4]) for r in range(0, len(shots), 4)])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    buf.tofile(os.path.join(work, "camera_sheet.jpg"))     # 한글 경로 - cv2.imwrite 는 조용히 실패한다
    print("camera_sheet.jpg")
