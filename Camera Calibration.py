import pyrealsense2 as rs
import numpy as np

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)

curr_frame = 0
while curr_frame < 1:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
        color_frame.profile)

    # print(depth_intrin.ppx, depth_intrin.ppy)
    # print(depth_intrin.fx, depth_intrin.fy)
    # print(depth_to_color_extrin.rotation)
    # print(depth_to_color_extrin.translation)

    mat = [1,2,3]
    rot = [[0.9999451637268066, 0.009572459384799004, -0.004250520374625921], [0.009596227668225765, 0.9999382495880127, -0.005607159808278084], [0.004196583293378353, 0.0056476411409676075, 0.9999752640724182]]
    tran = [1,1,1]
    fmat = [[depth_intrin.fx, 0, depth_intrin.ppx], [0, depth_intrin.fy, depth_intrin.ppy], [0, 0, 1]]
    # print(fmat)
    print(np.linalg.inv(fmat))
    print(np.linalg.inv(rot))
    temp = np.matmul(mat, np.linalg.inv(fmat))
    final = np.matmul(temp - tran, np.linalg.inv(rot))
    print(final)
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    curr_frame += 1
