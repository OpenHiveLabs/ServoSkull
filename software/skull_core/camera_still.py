"""Take one still with the skull camera and save it as a JPEG.

Usage:  python tools/bringup/camera_still.py [--out captures/] [--width 1536 --height 864]
"""
import argparse
from datetime import datetime, time
from pathlib import Path
from picamera2 import Picamera2

def timestamped_path(out_dir: Path) -> Path:
    """Return out_dir/still_YYYY-MM-DD_HH-MM-SS.jpg, creating out_dir if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    return out_dir / f"still_{datetime_str}.jpg"

def take_still(path: Path, width: int, height: int) -> None:
    """Configure the camera for a still of this size, capture, save, and close the camera."""
    camera = Picamera2()
    config = camera.create_still_configuration(main={"size": (width, height)})
    camera.configure(config)
    
    camera.start()
    time.sleep(2)
    camera.capture_file("TEST")
    camera.stop()
    raise NotImplementedError

def main() -> int:
    timestamped_path(Path("captures"))
    take_still(Path("captures/still.jpg"), 1536, 864)
    
    raise NotImplementedError

if __name__ == "__main__":
    raise SystemExit(main())