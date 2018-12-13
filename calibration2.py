import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
print("Environment Ready")
# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file('dataset4/realsense5/realsense.bag')            #("../object_detection.bag")
profile = pipe.start(cfg)

# Skip 1500 first frames to give the Auto-Exposure time to adjust
for x in range(10):
  pipe.wait_for_frames()
  
# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

color_image = np.asanyarray(color_frame.get_data())
depth_image=np.asanyarray(depth_frame.get_data())
print (color_image.shape)
print(depth_image.shape)
# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_color_frame=frameset.get_color_frame()
aligned_depth_frame = frameset.get_depth_frame()
colorizer = rs.colorizer()
aligned_color_depth_frame=(colorizer.colorize(aligned_depth_frame))
aligned_color_image=np.asanyarray(aligned_color_frame.get_data())
aligned_color_depth_image = np.asanyarray(aligned_color_depth_frame.get_data())
cv2.imshow('colorized depth',aligned_color_depth_image)
# Intrinsics & Extrinsics
depth_intrin = aligned_color_depth_frame.profile.as_video_stream_profile().intrinsics
color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
#depth_to_color_extrin = aligned_color_depth_frame.profile.get_extrinsics_to(aligned_color_frame.profile)
color_to_depth_extrin=aligned_color_frame.profile.get_extrinsics_to(aligned_color_depth_frame.profile)
# Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print('depth_scale:',depth_scale)
#depth_scale = depth_sensor.get_option(rs.option.depth_units)
# Map depth to color
color_pixel=[396,270]
pixel_distance_in_meters=aligned_depth_frame.get_distance(color_pixel[0],color_pixel[1])
#pixel_distance_in_meters=aligned_color_depth_frame.get_distance(color_pixel[0],color_pixel[1])
print("pixel distance in meters:",pixel_distance_in_meters)
color_point=rs.rs2_deproject_pixel_to_point(color_intrin,color_pixel,pixel_distance_in_meters)
depth_point=rs.rs2_transform_point_to_point(color_to_depth_extrin,color_point)
depth_pixel=rs.rs2_project_point_to_pixel(depth_intrin,depth_point)

pipe.stop()
print ('depth_intrinsic',depth_intrin)
print ('color_intrinsic',color_intrin)
print ('depth and color extrinsic',color_to_depth_extrin)
print('depth_pixel',depth_pixel)
print('depth_point',depth_point)
print('color_pixel',color_pixel)
print('color_point',color_point)
cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
#cv2.imshow('Align Example', images)
cv2.imshow('Align Example', aligned_color_image)
cv2.namedWindow('depth example', cv2.WINDOW_NORMAL)
cv2.imshow('depth example',aligned_color_depth_image)
cv2.waitKey(0)
