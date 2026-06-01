import os, sys, time, math, cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class VideoSystemProcessor:
    def __init__(self, input_path="data/input_video.y4m", width=352, height=288, fps=30, total_frames=30):
        self.input_path = input_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size_y = width * height
        self.frame_size_uv = (width // 2) * (height // 2) * 2
        self.frames_y = []
        self.residuals = []
        self.total_frames = total_frames 
        os.makedirs("data/output_videos", exist_ok=True)

    def prepare_mock_data(self):
        """Generates mock dynamic Y4M video file based on total_frames configured"""
        if not os.path.exists(self.input_path):
            print(f"INFO: Generating mock video data at {self.input_path}")
            with open(self.input_path, "wb") as f:
                f.write(f"YUV4MPEG2 W{self.width} H{self.height} F{self.fps}:1 Ip C420mpeg2\n".encode())
                # Đã cập nhật dòng này thành self.total_frames
                for i in range(self.total_frames):
                    f.write(b"FRAME\n")
                    y_data = np.full((self.height, self.width), 128, dtype=np.uint8)
                    cv2.circle(y_data, (60 + i*7, 144), 35 + (i % 6) * 3, 210, -1)
                    f.write(y_data.tobytes() + np.full((self.frame_size_uv,), 128, dtype=np.uint8).tobytes())

    def parse_raw_video(self):
        """Parses raw video input stream"""
        self.prepare_mock_data()
        with open(self.input_path, "rb") as f:
            first_line = f.readline()
            if b"YUV4MPEG2" in first_line:
                for token in first_line.decode('utf-8').split():
                    if token.startswith('W'): self.width = int(token[1:])
                    if token.startswith('H'): self.height = int(token[1:])
                self.frame_size_y = self.width * self.height
                while True:
                    if b"FRAME" not in f.readline(): break
                    y_bytes = f.read(self.frame_size_y)
                    if len(y_bytes) < self.frame_size_y: break
                    self.frames_y.append(np.frombuffer(y_bytes, dtype=np.uint8).reshape((self.height, self.width)))
                    f.read(self.frame_size_uv)
            else:
                f.seek(0)
                while True:
                    y_bytes = f.read(self.frame_size_y)
                    if len(y_bytes) < self.frame_size_y: break
                    self.frames_y.append(np.frombuffer(y_bytes, dtype=np.uint8).reshape((self.height, self.width)))
                    f.read(self.frame_size_uv)
        print(f"SUCCESS: Extracted {len(self.frames_y)} raw frames")
        return self.frames_y

    def extract_motion_residuals(self):
        """Extracts inter-frame residuals"""
        if not self.frames_y:
            self.parse_raw_video()
        self.residuals = [self.frames_y[i].astype(np.int16) - self.frames_y[i-1].astype(np.int16) for i in range(1, len(self.frames_y))]
        print(f"SUCCESS: Generated {len(self.residuals)} residual matrices")
        return self.residuals

print("STATUS: VideoSystemProcessor initialized")
