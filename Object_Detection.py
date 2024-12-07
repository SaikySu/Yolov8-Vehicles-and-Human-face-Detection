import os
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict

def process_video(video_path, face_model_path, vehicle_model_path, output_video_path, output_json_path):
    
    # Ensure program runs stably across different platforms
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'

    # Check if input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize models
    face_model = YOLO(face_model_path)
    vehicle_model = YOLO(vehicle_model_path)

    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video parameters
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Dictionary to store tracking information
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
    frame_count = 0

    print(f"Starting video processing: {os.path.basename(video_path)}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Face detection
        face_results = face_model.track(frame, persist=True)
        # Vehicle detection
        vehicle_results = vehicle_model.track(frame, persist=True)

        # Process face detection results
        if face_results[0].boxes.id is not None:
            boxes = face_results[0].boxes.xyxy.cpu().numpy()
            track_ids = face_results[0].boxes.id.cpu().numpy().astype(int)
            confidences = face_results[0].boxes.conf.cpu().numpy()

            for box, track_id, confidence in zip(boxes, track_ids, confidences):
                if confidence < 0.5:  # Threshold for face detection
                    continue

                class_name = 'face'
                if track_id not in objects[class_name]['tracks']:
                    objects[class_name]['count'] += 1
                    count = objects[class_name]['count']
                    objects[class_name]['tracks'][track_id] = {
                        'Object ID': f"{class_name.capitalize()} {count}",
                        'Class': class_name,
                        'Time_of_appearance': frame_count / fps,
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence)
                    }
                else:
                    objects[class_name]['tracks'][track_id].update({
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence)
                    })

                # Draw face detection
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = objects[class_name]['tracks'][track_id]['Object ID']
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Process vehicle detection results
        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.cpu().numpy().astype(int)
            classes = vehicle_results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = vehicle_results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                if confidence < 0.8:  # Threshold for vehicle detection
                    continue

                class_name = vehicle_model.names[cls]
                if class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                    continue

                if track_id not in objects[class_name]['tracks']:
                    objects[class_name]['count'] += 1
                    count = objects[class_name]['count']
                    objects[class_name]['tracks'][track_id] = {
                        'Object ID': f"{class_name.capitalize()} {count}",
                        'Class': class_name,
                        'Time_of_appearance': frame_count / fps,
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence)
                    }
                else:
                    objects[class_name]['tracks'][track_id].update({
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence)
                    })

                # Draw vehicle detection
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = objects[class_name]['tracks'][track_id]['Object ID']
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:  # Update progress every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing: {progress:.2f}% complete", end="")

    # Release resources
    cap.release()
    out.release()

    print(f"\nProcessed video saved at: {output_video_path}")

    # Save results to JSON
    json_results = []
    for class_name, data in objects.items():
        for track_id, info in data['tracks'].items():
            time_appeared = f"{int(info['Time_of_appearance'] // 60):02d}:{int(info['Time_of_appearance'] % 60):02d}"
            time_disappeared = f"{int(info['Time_of_disappearance'] // 60):02d}:{int(info['Time_of_disappearance'] % 60):02d}"

            json_results.append({
                "Object ID": info['Object ID'],
                "Class": info['Class'],
                "Time appeared": time_appeared,
                "Time disappeared": time_disappeared,
                "Bounding box": info['bounding_box'],
                "Confidence": f"{info['Confidence'] * 100:.2f}%"
            })

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)

    print(f"Tracking results saved at: {output_json_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video with face and vehicle detection")
    parser.add_argument('--input', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output_dir', type=str, default='./output', help="Directory to save results")
    parser.add_argument('--face_model', type=str, default='models/yolov8n-face.pt', help="Path to face detection model")
    parser.add_argument('--vehicle_model', type=str, default='models/yolov8m.pt', help="Path to vehicle detection model")
    
    args = parser.parse_args()

    # Create output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_video = os.path.join(args.output_dir, f"{base_name}_Out_Object.mp4")
    output_json = os.path.join(args.output_dir, f"{base_name}_Out_Object.json")

    # Process video
    process_video(
        video_path=args.input,
        face_model_path=args.face_model,
        vehicle_model_path=args.vehicle_model,
        output_video_path=output_video,
        output_json_path=output_json
    )