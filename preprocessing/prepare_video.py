"""Prepare subtitled frames in sequential 30-second clips."""

import argparse
import json

import cv2
from huggingface_hub import hf_hub_download
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def parse_srt_file(srt_path: Path) -> List[Dict[str, any]]:
    """
    Parse SRT subtitle file and return list of subtitle entries.
    
    Args:
        srt_path: Path to .srt file
        
    Returns:
        List of dicts with 'start', 'end', and 'text' keys
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match SRT entries
    # Format: number\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n
    pattern = r'(\d+)\n(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\n(.*?)(?=\n\d+\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    subtitles = []
    for match in matches:
        # Extract timestamp components
        start_h, start_m, start_s, start_ms = map(int, match[1:5])
        end_h, end_m, end_s, end_ms = map(int, match[5:9])
        
        # Convert to seconds
        start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
        end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
        
        # Extract text (remove speaker prefix if present, or keep it)
        text = match[9].strip()
        
        subtitles.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    
    return subtitles


def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    parts = time_str.split(',')
    time_part = parts[0].split(':')
    ms = int(parts[1]) if len(parts) > 1 else 0
    
    hours = int(time_part[0])
    minutes = int(time_part[1])
    seconds = int(time_part[2])
    
    return hours * 3600 + minutes * 60 + seconds + ms / 1000.0


def get_subtitle_at_time(subtitles: List[Dict], time_seconds: float) -> str:
    """Get the active subtitle text at a given time."""
    for sub in subtitles:
        if sub['start'] <= time_seconds <= sub['end']:
            return sub['text']
    return None


def wrap_text(text: str, font, font_scale: float, thickness: int, max_width: int) -> List[str]:
    """
    Wrap text into multiple lines that fit within max_width.
    """
    words = text.split(' ')
    lines = []
    current_line = ''
    
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines if lines else [text]


def draw_subtitle_on_frame(frame: np.ndarray, subtitle_text: str, 
                          font_size: int = 24, font_color: Tuple[int, int, int] = (255, 255, 255),
                          bg_color: Tuple[int, int, int] = (0, 0, 0),
                          position: str = 'bottom') -> np.ndarray:
    """
    Draw subtitle text on a frame.
    
    Args:
        frame: BGR image array
        subtitle_text: Text to display
        font_size: Font size
        font_color: BGR color tuple for text
        bg_color: BGR color tuple for background
        position: 'bottom' or 'center'
        
    Returns:
        Frame with subtitle drawn
    """
    if not subtitle_text:
        return frame
    
    height, width = frame.shape[:2]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size / 30.0
    thickness = max(1, int(font_size / 20))
    line_type = cv2.LINE_AA
    
    # Wrap text
    max_text_width = int(width * 0.9)
    text_lines = wrap_text(subtitle_text, font, font_scale, thickness, max_text_width)
    
    # Calculate dimensions
    line_heights = []
    line_widths = []
    for line in text_lines:
        (line_width, line_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(line_height + baseline)
        line_widths.append(line_width)
    
    # Box dimensions
    padding = 10
    line_spacing = 5
    box_width = max(line_widths) + 2 * padding
    box_height = sum(line_heights) + (len(text_lines) - 1) * line_spacing + 2 * padding
    
    # Position
    if position == 'bottom':
        y_pos = height - box_height - 20
    else:  # center
        y_pos = (height - box_height) // 2
    
    x_pos = (width - box_width) // 2
    
    # Draw background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_pos, y_pos),
        (x_pos + box_width, y_pos + box_height),
        bg_color,
        -1
    )
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw text
    text_x = x_pos + padding
    current_y = y_pos + padding + line_heights[0] if line_heights else y_pos + padding
    
    for i, line in enumerate(text_lines):
        line_width = line_widths[i]
        line_x = x_pos + (box_width - line_width) // 2
        
        cv2.putText(
            frame,
            line,
            (line_x, current_y),
            font,
            font_scale,
            font_color,
            thickness,
            line_type
        )
        
        if i < len(text_lines) - 1:
            current_y += line_heights[i] + line_spacing
    
    return frame


def process_video_with_subtitles(video_path: Path, srt_path: Path, 
                                  output_frames_dir: Path):
    """
    Process video: extract one frame per second and add subtitles.
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_frames_dir: Directory to save frames (e.g., data/frames/bedroom_01)
    """
    # Parse subtitles
    print(f"Parsing subtitle file: {srt_path}")
    subtitles = parse_srt_file(srt_path)
    print(f"Found {len(subtitles)} subtitle entries")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Video has no frame rate: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    
    print(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
    
    # Create output directory
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames - one per second, grouped into folders of 30 frames each
    # Track which seconds we've already extracted
    previous_second = -1
    frames_saved = 0
    frames_per_folder = 30
    
    print("\nExtracting frames with subtitles...")
    print(f"Grouping frames into folders of {frames_per_folder} frames each")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps if fps > 0 else 0.0
        current_second = int(current_time)
        
        # Extract one frame per second (only once per second)
        if current_second != previous_second:
            # Get subtitle for this time
            subtitle_text = get_subtitle_at_time(subtitles, current_time)
            
            # Draw subtitle on frame
            frame_with_subtitle = draw_subtitle_on_frame(
                frame.copy(),
                subtitle_text,
                font_size=28,
                font_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                position='bottom'
            )
            
            # Calculate folder number (1-indexed): every 30 frames go into a new folder
            folder_num = (frames_saved // frames_per_folder) + 1
            # Calculate frame number within folder (1-indexed)
            frame_num_in_folder = (frames_saved % frames_per_folder) + 1
            
            # Create directory for this folder (e.g., data/frames/bedroom_01/1/)
            folder_dir = output_frames_dir / str(folder_num)
            folder_dir.mkdir(parents=True, exist_ok=True)
            
            # Save frame (numbered within folder: 1.jpg, 2.jpg, ..., 30.jpg)
            frame_filename = folder_dir / f"{frame_num_in_folder}.jpg"
            if not cv2.imwrite(str(frame_filename), frame_with_subtitle):
                raise OSError(f"Could not write {frame_filename}")
            frames_saved += 1
            previous_second = current_second
            
            if frames_saved % 10 == 0:
                print(f"  Saved {frames_saved} frames (up to {current_second}s, folder {folder_num})")
        
        frame_count += 1
    
    cap.release()
    print(f"\n✓ Extracted {frames_saved} frames to {output_frames_dir}")
    if frames_saved == 0 or (total_frames > 0 and frame_count < total_frames):
        raise ValueError(f"Incomplete video: decoded {frame_count} of {total_frames} frames")
    return frames_saved


def prepare_video(video):
    questions = json.loads(Path("data/robot.json").read_text(encoding="utf-8"))
    if video not in questions:
        raise ValueError(f"Unknown Robot video: {video}")
    repo = "ByteDance-Seed/M3-Bench"
    # Older checkouts stored downloaded MP4s directly under data/videos.
    mp4 = Path("data/videos") / f"{video}.mp4"
    if not mp4.is_file():
        mp4 = Path(hf_hub_download(repo, f"videos/robot/{video}.mp4",
                                   repo_type="dataset", local_dir="data"))
    srt = Path("data/subtitles/robot") / f"{video}.srt"
    if not srt.is_file():
        hf_hub_download(repo, f"subtitles/robot/{video}.srt", repo_type="dataset", local_dir="data")
    process_video_with_subtitles(mp4, srt, Path("data/frames") / video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a single M3-Bench-robot video.")
    parser.add_argument("video")
    prepare_video(parser.parse_args().video)
