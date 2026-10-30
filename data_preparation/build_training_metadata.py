#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
CATEGORIES = ["haa500", "humman", "idea400", "kungfu", "music", "perform"]


def clip_base_name(seq_name: str) -> str:
    name = seq_name
    if name.startswith("video_"):
        name = name[len("video_"):]
    match = re.match(r"^(.*_clip\d+)", name)
    return match.group(1) if match else name


def flatten_caption(text: str) -> str:
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    text = re.sub(r"\.\s*", ", ", text)
    text = re.sub(r",\s*$", "", text)
    return text + "."


def load_motionx_caption(video: Path, video_root: Path, caption_root: Path,
                         flatten: bool) -> str | None:
    rel = video.relative_to(video_root)
    base = clip_base_name(video.stem)
    categories = [rel.parts[0]] if rel.parts else []
    categories += [c for c in CATEGORIES if c not in categories]
    for category in categories:
        path = caption_root / category / (base + ".txt")
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip().replace("\n", " ")
            return flatten_caption(text) if flatten else text
    return None



def load_captions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = {}
        for item in obj:
            if isinstance(item, dict) and "id" in item:
                out[str(item["id"])] = str(
                    item.get("motion_caption", item.get("caption", item.get("prompt", "")))
                )
        return out
    raise ValueError("captions JSON must be an object or a list of records")


def caption_for(video: Path, video_root: Path, captions: dict[str, str],
                 caption_root: Path, flatten: bool, default: str | None) -> str:
    rel = video.relative_to(video_root)
    keys = [str(rel), rel.stem, video.stem]
    for key in keys:
        if key in captions and captions[key].strip():
            text = captions[key].strip()
            return flatten_caption(text) if flatten else text
    text = load_motionx_caption(video, video_root, caption_root, flatten)
    if text:
        return text
    if default is not None:
        return default
    raise ValueError(
        f"missing caption for {rel}; expected Motion-X++ text/semantic_label "
        f"or provide --captions/--default-caption"
    )


def find_condition(root: Path, rel: Path) -> Path | None:
    candidates = [
        root / rel,
        root / rel.parent / f"{rel.stem}.mp4",
        root / rel.parent / f"{rel.stem}.mov",
        root / rel.parent / f"{rel.stem}.webm",
    ]
    return next((p for p in candidates if p.is_file()), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--mesh-root", type=Path, required=True)
    ap.add_argument("--depth-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--captions", type=Path)
    ap.add_argument("--caption-root", type=Path,
                    help="Motion-X++ text/semantic_label root; defaults beside video-root")
    ap.add_argument("--flatten-caption", action="store_true",
                    help="merge internal sentence breaks into one caption")
    ap.add_argument("--default-caption")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip samples with missing mesh/depth instead of failing")
    args = ap.parse_args()

    video_root = args.video_root.resolve()
    mesh_root = args.mesh_root.resolve()
    depth_root = args.depth_root.resolve()
    captions = load_captions(args.captions)
    caption_root = (args.caption_root or (video_root.parent / "text" / "semantic_label")).resolve()
    videos = sorted(p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    records = []
    skipped = 0
    output_base = args.output.resolve().parent

    for video in videos:
        rel = video.relative_to(video_root)
        mesh = find_condition(mesh_root, rel)
        depth = find_condition(depth_root, rel)
        if mesh is None or depth is None:
            if args.allow_missing:
                skipped += 1
                continue
            raise FileNotFoundError(f"missing condition for {rel}: mesh={mesh}, depth={depth}")
        record = {
            "video_path": str(video.resolve().relative_to(output_base))
            if video.resolve().is_relative_to(output_base) else str(video.resolve()),
            "mesh_path": str(mesh.resolve().relative_to(output_base))
            if mesh.resolve().is_relative_to(output_base) else str(mesh.resolve()),
            "depth_path": str(depth.resolve().relative_to(output_base))
            if depth.resolve().is_relative_to(output_base) else str(depth.resolve()),
            "motion_caption": caption_for(
                video, video_root, captions, caption_root,
                args.flatten_caption, args.default_caption
            ),
        }
        records.append(record)
        if args.limit > 0 and len(records) >= args.limit:
            break

    if not records:
        raise RuntimeError("no complete video/mesh/depth records found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} record(s) to {args.output}; skipped={skipped}")


if __name__ == "__main__":
    main()