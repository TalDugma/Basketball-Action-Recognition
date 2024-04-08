from ultralytics import YOLO
import torch
from tqdm import tqdm
import numpy as np
import os

def sum_distances(tensor, center_point=(88, 64)):
    sum_distance = 0
    counter = 0.0
    for point in tensor:
        if point[0] != 0 and point[1] != 0:
            counter += 1
        else:
            sum_distance += ((point[0] - center_point[0])**2 + (point[1] - center_point[1])**2)**0.5
    return sum_distance / counter if counter else 0

def create_skeletons(args):

    if args.no_gpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load a model and transfer it to the specified device
    model = YOLO(args.pose_model_name, verbose=False).to(device)  

    # Process the videos in dataset/example folder and save the output in skeletal_dataset folder
    videos_folder_path = args.data
    skeletal_dataset_path = "skeletal_dataset"
    os.makedirs(skeletal_dataset_path, exist_ok=True)

    for video in tqdm(os.listdir(videos_folder_path)):
        if video.endswith(".mp4"):
            list_of_tensors = []
            video_path = os.path.join(videos_folder_path, video)
            results = model(source=video_path, show=False, verbose=False, stream=True)
            for result in results:
                if len(result.boxes) > 1:
                    min_distance = np.inf
                    min_index = -1
                    for i, box in enumerate(result.boxes):
                        distance = sum_distances(result.keypoints[i].xy.squeeze(0).cpu().numpy())
                        if distance < min_distance:
                            min_distance = distance
                            min_index = i
                    tensor = result.keypoints[min_index].xy.squeeze(0)
                elif len(result.boxes) == 0:
                    continue
                else:
                    tensor = result.keypoints.xy.squeeze(0)
                list_of_tensors.append(tensor)
            if list_of_tensors:
                torch.save(torch.stack(list_of_tensors), os.path.join(skeletal_dataset_path, video[:-4] + ".pt"))

