import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("REMOVE_ROOT") or os.path.abspath(os.path.join(_HERE, ".."))


def _p(env, *default):
    return os.environ.get(env) or os.path.join(*default)


# These two projects are installed into ReMoVe/tools by setup/clone_tools.sh.
GVHMR_ROOT = os.environ.get("GVHMR_ROOT", os.path.join(ROOT, "tools", "GVHMR"))
SMIRK_DIR = os.environ.get("SMIRK_DIR", os.path.join(ROOT, "tools", "SMIRK"))
DECA_DIR = _p("DECA_DIR", ROOT, "assets", "DECA")
HY_ROOT = os.environ.get("HY_ROOT", "")  # legacy (HY-Motion), not needed any more

# weights
WEIGHTS = _p("REMOVE_WEIGHTS", ROOT, "weights")
BODY_MODELS = _p("BODY_MODELS", GVHMR_ROOT, "inputs", "checkpoints", "body_models")
SMPL_PATH = _p("SMPL_PATH", BODY_MODELS, "smpl", "SMPL_NEUTRAL.pkl")
SMPLX_DIR = _p("SMPLX_DIR", BODY_MODELS, "smplx")
SMPLX_PATH = _p("SMPLX_PATH", SMPLX_DIR, "SMPLX_NEUTRAL.npz")
SAM3_CKPT = _p("SAM3_CKPT", WEIGHTS, "sam3", "sam3.pt")
SAM3_BPE = _p("SAM3_BPE", WEIGHTS, "sam3", "bpe_simple_vocab_16e6.txt.gz")
SMIRK_CKPT = _p("SMIRK_CKPT", SMIRK_DIR, "pretrained_models", "SMIRK_em1.pt")
FACE_LANDMARKER = _p("FACE_LANDMARKER", WEIGHTS, "mediapipe", "face_landmarker.task")
HEAD_TEMPLATE_OBJ = _p("HEAD_TEMPLATE_OBJ", DECA_DIR, "data", "head_template.obj")
# step 9: LoRA + expanded patch-embedding weights of the mesh+depth conditioned Wan2.2-TI2V-5B
WAN_LORA = _p("WAN_LORA", WEIGHTS, "wan_lora", "Wan2.2-TI2V-5B-remove_step-31110.safetensors")

if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.isupper() and isinstance(v, str):
            print(f"{k:18s} {'OK ' if os.path.exists(v) else 'MISSING'}  {v}")

