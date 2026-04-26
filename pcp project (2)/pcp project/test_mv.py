import av
import numpy as np

container = av.open('test_video_h264.mp4')
stream = container.streams.video[0]
stream.codec_context.options = {'flags2': '+export_mvs'}

for i, frame in enumerate(container.decode(stream)):
    mvs = frame.side_data.get('MOTION_VECTORS')
    if mvs:
        mv_arr = mvs.to_ndarray()
        print(f"Frame {i}: Found {len(mv_arr)} MVs. Shape: {mv_arr.shape}")
        print("Sample MVs (source_x, source_y, dst_x, dst_y, motion_x, motion_y, ...):")
        print(mv_arr[0:2])
        break
    if i > 50:
        print("No MVs found in first 50 frames")
        break
