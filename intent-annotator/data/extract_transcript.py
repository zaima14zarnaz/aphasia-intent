import json
import os

SOURCE_DIR = "/data/quantization/zaima/aphasiabank/neural2/control/ts_videos/video_anns"
DEST_FILE = "/home/z/zaimazarnaz/research1/Aphasia/intent-annotator/data/clips.json"

def main():
    with open(DEST_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for clip in data["clips"]:
        video_filename, gesture_idx, _ = clip["filename"].split('_')
        
        metadata_path = os.path.join(SOURCE_DIR, f"{video_filename}_metadata.json")
        if not os.path.exists(metadata_path):
            continue
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        for b in metadata["gestures"]:
            if b["gesture_index"] == int(gesture_idx):
                clip["task"] = b["task"]
                clean_lines = [
                    line for line in b["transcript_lines"] 
                    if not line.strip().lower().startswith("marker:")
                ]
                clip["transcript"] = "\n".join(clean_lines)
                clip["placeholder_context"] = False
                break

    with open(DEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print("task success") 

if __name__ == "__main__":
    main()