import pyrealsense2 as rs
import cv2, math
import numpy as np
import os
import time
from PIL import Image
import torch
import pandas as pd

def main() :
    # Configure depth and color streams
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
        
    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    COLORS = [[0, 0, 255]]

    # Load the model
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    model.conf = 0.1

    # Get pandas frame to save the confidence of the center pixel detection and its class
    pred_prob = pd.DataFrame(columns=['confidence', 'class'])
    frame_seq = -1
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        frame_seq += 1

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
            
        image = Image.fromarray(color_image)
        img = np.asarray(image)
       
        # Inference
        results = model(img)

        # Get the detection in the certer pixel if any
        print(results.pandas().xyxy[0])
        found_bottle = False
        # Iterate pandas rows
        for index, row in results.pandas().xyxy[0].iterrows():
            # Check if there is a bottle detection
            # Headers are ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
            if row['name'] == 'bottle' or row['name'] == 'tv':
                if row['name'] == 'bottle':
                    found_bottle = True
                # Save the confidence, the class and depth of the bottom pixel
                print("Save reading")
                bottle_bottom = np.array([row['xmin'], row['ymin']])
                depth = depth_image[int(bottle_bottom[1]), int(bottle_bottom[0])]
                # if depth <= 0:
                #     continue
                pred_prob = pred_prob.append({'confidence': row['confidence'], 'class': row['name'], 'depth': frame_seq}, ignore_index=True)
                # Save dataframe
                pred_prob.to_csv('pred_prob.csv', index=False)
            
        if not found_bottle:
            pred_prob = pred_prob.append({'confidence': 0, 'class': 'bottle', 'depth': frame_seq}, ignore_index=True)
            # Save dataframe
            pred_prob.to_csv('pred_prob.csv', index=False)

        # Results
        results_cv = results.render()[0]
        # results_cv = cv2.cvtColor(results_cv, cv2.COLOR_RGB2BGR)

        cv2.imshow("Image", results_cv)

        #Save image
        cv2.imwrite(f'./nn_characterization/images/orient/{frame_seq:05}.png', results_cv)

        #cv2.imshow("Depth", depth_image_ocv)
            
        cv2.waitKey(1)


    cv2.destroyAllWindows()

    print("\nFINISH")

if __name__ == "__main__":
    main()