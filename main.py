from LSTM_Attention import train_classifier
from skeletal_extraction import create_skeletons
from FullPipeline import infer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Infer the model')
    parser.add_argument('--model', type=str, help='Model to use', default="trained_model.pth")
    parser.add_argument('--video', type=str, help='Video to use', default="videos/Knicks3pointer.mp4")
    parser.add_argument('--extract_data', action='store_true', help='Extract skeletons from dataset')
    parser.add_argument('--data', type=str, help='Dataset directory', default="dataset/examples")
    parser.add_argument('--output', type=str, help='Output directory', default="output")
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU if not available')
    parser.add_argument('--seed', type=int, help='Seed for random number generation', default=42)
    parser.add_argument('--input_size', type=int, help='Input size', default=34)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=2)
    parser.add_argument('--pose_model_name', type=str, help='Skeleton extractor model name', default="yolov8n-pose.pt")
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    if args.train == True:
        if args.extract_data:
            create_skeletons(args)
        train_classifier(args)
    elif args.infer:
        infer(args)
    else:
        print("Please specify either train or infer")

