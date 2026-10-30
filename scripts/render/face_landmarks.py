import argparse, os
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
from paths import FACE_LANDMARKER
DEFAULT_MODEL=FACE_LANDMARKER

def make(model):
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model), num_faces=1,
        running_mode=vision.RunningMode.IMAGE))

def _run(lmk, rgb):
    mi=mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    r=lmk.detect(mi)
    return r.face_landmarks[0] if r.face_landmarks else None

def detect(path, lmk):
    img=cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: print(f"[skip] {path}"); return None
    if img.ndim==3 and img.shape[2]==4: img=img[:,:,:3]
    if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    H,W=img.shape[:2]; rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # 1) full image
    lm=_run(lmk,rgb)
    if lm is not None:
        return np.array([[p.x*W,p.y*H] for p in lm],np.float32)
    # 2) upper-centre crop search, map coords back to full image
    for (y0,y1,x0,x1) in [(0,0.55,0.2,0.8),(0,0.4,0.25,0.75),(0.05,0.45,0.3,0.7),
                          (0,0.6,0.15,0.85),(0,0.5,0.0,1.0),(0,0.35,0.35,0.65)]:
        Y0,Y1,X0,X1=int(y0*H),int(y1*H),int(x0*W),int(x1*W)
        c=rgb[Y0:Y1,X0:X1]
        if c.size==0: continue
        sc=640.0/max(c.shape[0],1); cc=cv2.resize(c,None,fx=sc,fy=sc)
        lm=_run(lmk,cc)
        if lm is not None:
            ch,cw=cc.shape[:2]
            return np.array([[X0+p.x*cw/sc, Y0+p.y*ch/sc] for p in lm],np.float32)
    print(f"[nolmk] {path}"); return None

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("images",nargs="+")
    ap.add_argument("--out_dir",required=True); ap.add_argument("--model",default=DEFAULT_MODEL)
    a=ap.parse_args(); os.makedirs(a.out_dir,exist_ok=True); lmk=make(a.model)
    for p in a.images:
        pts=detect(p,lmk)
        if pts is None: continue
        stem=os.path.splitext(os.path.basename(p))[0]
        out=os.path.join(a.out_dir,stem+".mplmk.npy"); np.save(out,pts)
        print(f"[ok] {os.path.basename(p)} -> {out} ({pts.shape[0]} pts)")
