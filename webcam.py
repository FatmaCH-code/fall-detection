"""Detection de chutes en temps reel (webcam ou video). Q pour quitter.
Usage :  python webcam.py            ou   python webcam.py --video test.mp4
Fichiers requis dans le meme dossier : best_model_v2.pth, config.json"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn

# 17 keypoints COCO, dans l'ordre exact utilise par le modele
KP_NAMES = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle']
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

T = 120               # nombre de frames par sequence
FPS_ASSUMED = 30      # le dataset est a 30 fps -> 120 frames = 4 secondes
WINDOW_SEC = T / FPS_ASSUMED


def normalize_seq(p):
    """p : (T, 17, 2) en coordonnees IMAGE (y vers le bas) -> (T, 34).
    Centre sur les hanches de la 1ere frame, divise par la taille du torse."""
    p = np.asarray(p, dtype=np.float32)
    hips = p[:, 11:13, :].mean(1)
    shoulders = p[:, 5:7, :].mean(1)
    torso = np.linalg.norm(shoulders - hips, axis=1)
    scale = max(float(np.median(torso)), 1.0)
    out = (p - hips[0]) / scale
    return np.clip(out, -10, 10).reshape(len(p), -1).astype(np.float32)


class FallTransformer(nn.Module):
    def __init__(self, input_dim=34, num_heads=2, num_layers=2, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        # encodage de position appris : indique au modele l'ORDRE des frames
        self.pos = nn.Parameter(torch.randn(1, T, hidden_dim) * 0.02)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, x):
        x = self.embedding(x) + self.pos[:, :x.size(1)]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


import argparse, csv, datetime, json, os, time, urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, 'best_model_v2.pth')
CONFIG_FILE = os.path.join(HERE, 'config.json')
POSE_FILE = os.path.join(HERE, 'pose_landmarker.task')
LOG_FILE = os.path.join(HERE, 'falls_log.csv')
# indices MediaPipe -> 17 keypoints COCO, meme ordre que KP_NAMES
KP_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
CONFIRM = 2          # nombre d'analyses consecutives positives avant l'alarme

ap = argparse.ArgumentParser()
ap.add_argument('--video', help="chemin d'une video a analyser (sinon webcam)")
ap.add_argument('--camera', type=int, default=0)
ap.add_argument('--threshold', type=float, help='seuil de chute (0-1), sinon config.json')
args = ap.parse_args()

cfg = {'THRESHOLD': 0.8}
if os.path.exists(CONFIG_FILE):
    cfg.update(json.load(open(CONFIG_FILE)))
THRESHOLD = args.threshold if args.threshold is not None else cfg['THRESHOLD']

model = FallTransformer()
state = torch.load(MODEL_FILE, map_location='cpu')
if 'pos' not in state:      # ancien modele entraine sans encodage de position
    res = model.load_state_dict(state, strict=False)
    assert res.missing_keys == ['pos'] and not res.unexpected_keys, res
    model.pos.data.zero_()  # pos = 0 -> comportement identique a l'ancien modele
    print("Ancien modele (sans encodage de position) charge")
else:
    model.load_state_dict(state)
model.eval()

if not os.path.exists(POSE_FILE):
    print('Telechargement du modele MediaPipe...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_lite/float16/1/pose_landmarker_lite.task', POSE_FILE)
detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=POSE_FILE)))


def beep():
    try:
        import winsound
        winsound.Beep(1200, 400)
    except Exception:
        print('\a', end='')


def predict(hist, now):
    """Re-echantillonne les WINDOW_SEC dernieres secondes sur T frames et predit."""
    ts = np.array([h[0] for h in hist])
    flat = np.stack([h[1] for h in hist]).reshape(len(hist), -1)
    target = np.linspace(now - WINDOW_SEC, now, T)
    seq = np.stack([np.interp(target, ts, flat[:, k]) for k in range(34)], 1).reshape(T, 17, 2)
    with torch.no_grad():
        x = torch.from_numpy(normalize_seq(seq)).unsqueeze(0)
        return torch.softmax(model(x), 1)[0, 1].item()


cap = cv2.VideoCapture(args.video if args.video else args.camera)
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
hist, prob, positives, n, last_alarm, frame_i = deque(), 0.0, 0, 0, 0.0, 0
print('Q pour quitter | seuil =', round(THRESHOLD, 2))

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    now = (frame_i / fps_video) if args.video else time.time()
    frame_i += 1
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    status, color = 'Pas de personne', (255, 255, 0)
    if res.pose_landmarks:
        lms = res.pose_landmarks[0]
        pts = np.array([[lms[i].x * w, lms[i].y * h] for i in KP_IDX], dtype=np.float32)
        for x_, y_ in pts:
            cv2.circle(frame, (int(x_), int(y_)), 4, (0, 255, 255), -1)
        hist.append((now, pts))
        while hist and now - hist[0][0] > WINDOW_SEC + 1:
            hist.popleft()
        n += 1
        if now - hist[0][0] < WINDOW_SEC:
            status, color = 'Analyse...', (255, 255, 0)
        else:
            if n % 5 == 0:
                prob = predict(hist, now)
                positives = positives + 1 if prob > THRESHOLD else 0
            if positives >= CONFIRM:
                status, color = f'CHUTE DETECTEE ({prob * 100:.0f}%)', (0, 0, 255)
                if now - last_alarm > 10:
                    last_alarm = now
                    beep()
                    new = not os.path.exists(LOG_FILE)
                    with open(LOG_FILE, 'a', newline='') as fh:
                        wr = csv.writer(fh)
                        if new:
                            wr.writerow(['heure', 'probabilite'])
                        wr.writerow([datetime.datetime.now().isoformat(timespec='seconds'), round(prob, 3)])
            else:
                status, color = f'Normal (risque chute: {prob * 100:.0f}%)', (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow('Fall Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
