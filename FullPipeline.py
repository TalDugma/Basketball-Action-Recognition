import cv2
import torch
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
import supervision as sv
from collections import defaultdict
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os

labels_dict = {0 : "block", 1 : "pass", 2 : "run", 3: "dribble",4: "shoot",5 : "ball in hand", 6 : "defense", 7: "no_action" , 8 : "no_action" , 9: "walk" ,10: "discard"}

def sum_distances(tensor, center_point):
    sum_distance = 0
    counter = 0.0
    for point in tensor:
        if point[0] != 0 and point[1] != 0:
            counter += 1
        else:
            sum_distance += ((point[0] - center_point[0])*2 + (point[1] - center_point[1])*2)*0.5
    return sum_distance / counter if counter else 0



def is_in_court(bounding_box_person, court_mask,frame_num):
    #the bounding box is in the format (x1,y1,w,h)
    #the mask is a binary mask
    
    x, y, w, h = bounding_box_person[0]
    
    
    # check if the bottom of the bounding box is in the court
    if np.all(court_mask[y-int(h/2):y+int(h/2),x-int(w/2)] == 0):
        return False
    return True
  
def get_court_bounding_boxes(video,CLIENT):
    court_bounding_boxes = []
    for frame in video:
        result = CLIENT.infer(frame, model_id="court-detection/1")
        detections = sv.Detections.from_inference(result)
        
        try:
            mask_court = detections.mask[1]
            mask_three_point = detections.mask[0]
            #final mask is union of masks
            final_mask = mask_court + mask_three_point 
        except: 
            try:
                final_mask = court_bounding_boxes[-1]
            except:
                final_mask = np.ones((frame.shape[0],frame.shape[1]))
        court_bounding_boxes.append(final_mask)

    #for each mask print how many pixels are in the court
    # for i, mask in enumerate(court_bounding_boxes):
    #     print(f"frame{i},pixels indentified:{mask.sum()}")
    return court_bounding_boxes

def get_person_bounding_boxes(video_path, person_detection_model):
    """
    Get the bounding boxes of the people in the video  : format {frame_number: [ordered list of bounding boxes(xyxy format)]}  
    """
    model = person_detection_model
    results = model.track(source=video_path, tracker="bytetrack.yaml",persist=True,conf=0.1,show=False)  # Tracking with ByteTrack tracker
    video = cv2.VideoCapture(video_path)
    entities_list_per_frame = []
    for result in results:
        bbox_dict = defaultdict(list)
        boxes = result.boxes
        xywhs = boxes.xywh
        if boxes.id != None:
            for i,xywh in enumerate(xywhs):
                x1,y1,w,h = int(xywh[0]),int(xywh[1]),int(xywh[2]),int(xywh[3])
                bbox_dict[int(boxes.id[i])].append((x1,y1,w,h))
        entities_list_per_frame.append(bbox_dict)
    return entities_list_per_frame

def players_through_vid(person_bounding_boxes,video):
    players_through_vid = defaultdict(list)
    players_through_vid_idx = defaultdict(list)
    for i, frame in enumerate(video):
        persons_idetified = person_bounding_boxes[i]
        for person in persons_idetified.keys():
            x, y, w, h = persons_idetified[person][0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            person_frame = frame[y-int(h/2):y+int(h/2), x-int(w/2):x+int(w/2)]
            # pass the person_frame to the classification model
            players_through_vid[person].append([i, "end of frame", "1.0", (x,y,w,h), person_frame])

            # get the label
            # append the label to the players_through_vid
       
    return players_through_vid

def get_skeleton_features(player_vid,extrcting_model,center_point):

    # Extract the skeleton features
    results = extrcting_model(source=player_vid, show=False, verbose=False, stream=True)
    list_of_tensors = []
    for result in results:
        if len(result.boxes) > 1:
            min_distance = np.inf
            min_index = -1
            for i, box in enumerate(result.boxes):
                cur_center_point = (center_point[i][0],center_point[i][1])
                distance = sum_distances(result.keypoints[i].xy.squeeze(0).cpu().numpy(),center_point=cur_center_point)
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            tensor = result.keypoints[min_index].xy.squeeze(0)
        elif len(result.boxes) == 0:
            continue
        else:
            tensor = result.keypoints.xy.squeeze(0)
        list_of_tensors.append(tensor)
    return list_of_tensors

def draw_bounding_boxes(frame, bounding_box, label, confidence):
    # Unpack the bounding box coordinates
    x, y, w, h = bounding_box
    # Draw the bounding box rectangle
    cv2.rectangle(frame, (x - int(w / 2), y - int(h / 2)), (x + int(w / 2), y + int(h / 2)), (0, 255, 0), 2)
    
    # Prepare the label text with confidence score
    label_text = f'{label}: {confidence:.2f}'
    # Calculate text width & height to background the text for visibility
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # Draw background rectangle for text for better visibility
    cv2.rectangle(frame, (x - int(w / 2), y - int(h / 2) - 20), (x - int(w / 2) + text_width, y - int(h / 2)), (0, 255, 0), -1)
    # Put the label text on the frame
    cv2.putText(frame, label_text, (x - int(w / 2), y - int(h / 2) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_length, hidden_size]
        attention_weights = torch.softmax(self.attention(lstm_output).squeeze(2), dim=1)
        # attention_weights shape: [batch_size, seq_length]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        # context_vector shape: [batch_size, hidden_size]
        return context_vector, attention_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        context_vector, attention_weights = self.attention(lstm_out)
        out = self.fc(context_vector)
        return out

def infer(args):

    os.makedirs(args.output, exist_ok=True)

    print("Starting the pipeline.")
    print("saving outputs to",args.output)
    print("Loading the classification extrcting_model.")
    # load the saved model (saved as ordered dict)

    if args.no_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classification_model = LSTMWithAttention(input_size=34, hidden_size=126, num_layers=2, num_classes=8,device= device).to(device)
    classification_model.load_state_dict(torch.load(args.model))
    classification_model.eval()
    # load the saved model (saved as ordered dict)

    # ball_detection_model = YOLO('yolov8n.pt') # Load the ball detection model
    
    print("model loaded.")

    print("Preprocessing the video.")
    # Load the inference video
    video_path = args.video
    frames = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)

    cap.release()
    video = frames
    print("Preprocessing complete.")

    print("Segmenting the court")
    # Get the court bounding boxes
    CLIENT = InferenceHTTPClient(
        api_url="https://outline.roboflow.com",
        api_key="TU495Zl1i6A9iU5HLQU6"
    )   
    # Load the court segmentation model
    court_bounding_boxes = get_court_bounding_boxes(video, CLIENT)
    print("Segmentation complete.")

    print("Detecting players in the video.")
    # Get the person bounding boxes
    person_detection_model = YOLO('yolov8x.pt')  # Load the person detection model
    person_bounding_boxes = get_person_bounding_boxes(video_path, person_detection_model) #List of lists [frame1: [person1, person2,...], frame2: [person1, person2]
    print("Detection complete.")

    print("Removing the players that are not in the court.")
    # Remove from the list of person bounding boxes the ones that are not in the court.
    for i, frame in enumerate(video):
        persons_idetified = person_bounding_boxes[i]
        court_in_cur_frame = court_bounding_boxes[i]
        list_to_remove = []
        for person in persons_idetified.keys():
            
            if not is_in_court(persons_idetified[person], court_in_cur_frame,frame_num=i):
                list_to_remove.append(person)
        for person in list_to_remove:
            del persons_idetified[person]
    print(f"Removal complete.,{len(list_to_remove)} players removed.")

    # Extract the skeleton features
    extracting_model = YOLO('yolov8x-pose.pt', verbose=False)


    players_through_v = players_through_vid(person_bounding_boxes,video)


    print("Identifying the players action.")
    # Generate labels for identified - in court - persons
    player_labels = defaultdict(list)
    to_delete = []
    for player in tqdm(players_through_v.keys()):
        try:
            player_frames = [X[4] for X in players_through_v[player]]
            # get the center point of the player
            center_point = []
            for frame in player_frames:
                center_point.append([frame.shape[1]//2,frame.shape[0]//2])
            
            with torch.no_grad():
                skeletons = get_skeleton_features(player_frames,extrcting_model=extracting_model, center_point=center_point)
                labels=[]
                confidences=[]
                #split skeletons to batches of 5 (if last batch is less than 5, save it as is)
                for i in range(0,len(skeletons),16):
                    if i+16 >= len(skeletons):
                        cur_window = skeletons[i:]
                    else:
                        cur_window = skeletons[i:i+16]
                    cur_window_stacked = torch.stack(cur_window)
                    cur_window_to_model = cur_window_stacked.view(cur_window_stacked.size(0),-1).unsqueeze(0)
                    cur_window_to_model = cur_window_to_model.to(device)
                    outputs = classification_model(cur_window_to_model)
                    outputs = torch.softmax(outputs, dim=1)
                    lab = torch.argmax(outputs,dim=1)
                    label = labels_dict[lab.item()]
                    conf = torch.max(outputs, dim=1)[0]

                    lab =[label] * len(cur_window)
                    conf = [conf] * len(cur_window)
                    labels.extend(lab)
                    confidences.extend(conf)

            for j, frame in enumerate(players_through_v[player]):
                if j >= len(labels):
                    pass
                else:
                    players_through_v[player][j][1] = labels[j]
                    players_through_v[player][j][2] = confidences[j]
                
        except:
            to_delete.append(player)
    
    for player in to_delete:
        del players_through_v[player]

    print("Identification complete.")

    print("Generating the output video.")
    # Generate the output video
    output_video = []
    for i, frame in enumerate(video):
        for person in person_bounding_boxes[i].keys():
            for frame_detailes in players_through_v[person]:
                idx = frame_detailes[0]
                label = frame_detailes[1]
                confidence = float(frame_detailes[2])
                bounding_box = frame_detailes[3]
                if idx == i:
                    frame = video[idx]
                    frame = draw_bounding_boxes(frame, bounding_box, label ,confidence)
        output_video.append(frame)
    print("Output video generation complete.")
    
    print("Saving the output video.")
    # Save the output video
    output_path = f"{args.output}/output_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
    for frame in output_video:
        out.write(frame)
    out.release()
    
    print(f"Output video saved at {output_path}.")
    print("Pipeline complete.")