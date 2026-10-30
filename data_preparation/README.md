# Training-condition preparation

This directory contains the public preprocessing code used to construct ReMoVe's
mesh and depth conditioning videos. The Motion-X++ videos and annotations are not
included in this repository.

Expected dataset layout:

~~~text
<MOTIONX_ROOT>/
 video/<category>/<clip>.mp4
 motion/mesh_recovery/global_motion/<category>/<clip>.json
~~~

## Motion-X++ categories

The released preprocessing pipeline currently supports these six Motion-X++ categories:

| Motion-X++ annotation category | Video directory | Notes |
|---|---|---|
| `haa500` | `video/haa500` | HAA500 actions |
| `idea400` | `video/idea400_light` | the annotation name is `idea400`, but videos are under `idea400_light` |
| `humman` | `video/humman` | keep the dataset spelling `humman` |
| `kungfu` | `video/kungfu` | kungfu motions |
| `music` | `video/music` | music motions |
| `perform` | `video/perform` | performance motions |

For full preprocessing, download the videos and the matching
`motion/mesh_recovery/global_motion/<category>/` annotations for all six
categories. For a smoke test, one category and a small subset of matching clips
is enough; the annotation JSON stem and video filename stem must match.

If captions are needed, the default Motion-X++ caption lookup expects semantic
labels under:

```text
<MOTIONX_ROOT>/text/semantic_label/<category>/<clip>.txt
```

The preprocessing scripts do not redistribute Motion-X++ data. Users should
download the required categories from the official Motion-X++ release and set
`MOTIONX_ROOT` to the resulting local directory.

The public launcher writes the generated conditions to mesh/, person_masks/,
and depth/ under the dataset root by default:

~~~bash
export GVHMR_ROOT=/path/to/GVHMR
export SAM3_CHECKPOINT=/path/to/sam3.pt

bash data_preparation/prepare_training_conditions.sh /data/MotionX++ 8
~~~

All preprocessing stages use the single ReMoVe environment.


## Building the training metadata

After the three condition directories exist, build the JSON consumed by the
training launcher. Use --limit for a smoke test; this command only scans paths
and does not read or generate video frames.

~~~bash
python data_preparation/build_training_metadata.py   --video-root /data/MotionX++/video   --mesh-root /data/MotionX++/mesh   --depth-root /data/MotionX++/depth   --captions /data/MotionX++/captions.json   --output /data/MotionX++/data.json

# smoke test:
python data_preparation/build_training_metadata.py   --video-root /data/MotionX++/video   --mesh-root /data/MotionX++/mesh   --depth-root /data/MotionX++/depth   --default-caption "A person performing human motion."   --output /tmp/remove-data.json   --limit 3
~~~
