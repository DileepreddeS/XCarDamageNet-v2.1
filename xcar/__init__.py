"""XCarDamageNet — YOLO11m with physics, attention and contrastive aux modules.

Every extension is implemented as a subclass; ultralytics itself is unmodified.
"""

__version__ = "2.1.0"
ULTRALYTICS_PIN = "8.4.48"

CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
NUM_CLASSES = len(CLASS_NAMES)
