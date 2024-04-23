from analyze_video import analyze_video
import os, cv2, shutil

def convert_video_to_frames(video_path, save_dir):
    # Make sure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Determine video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        # Read a frame
        success, frame = cap.read()

        # If the frame was not successfully read, break the loop
        if not success:
            break

        # Save every frame without trying to simulate 30 FPS
        save_path = os.path.join(save_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(save_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'All frames have been saved to {save_dir}. Total frames: {frame_count}.')
    return original_fps

# Path to the directory where the frames will be saved
save_dir = 'frames'

# Path to the directory containing the video
video_path = 'videos/test.mp4'

convert_video_to_frames(video_path, save_dir)

figures, jsonData = analyze_video(save_dir)

shutil.rmtree(save_dir)

attention = 50
for c, fig in figures.items():
    fig.savefig(f"output/results_{attention}%_attention.png")
    attention += 50
