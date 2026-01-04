# Optimized Threat Detection System
import cv2
import torch
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
@dataclass
class Config:
    SOURCE: str = r"background\bg_432.mp4"
    MODEL_PATH: str = "yolov8l-pose.pt"
    CONF_THRES: float = 0.5
    IMG_SIZE: int = 640
    
    # Detection thresholds
    VEL_THRESH: int = 100
    PROXIMITY_THRESH: float = 0.25
    SMOOTHING_WINDOW: int = 3
    
    # Quality filters
    MIN_KEYPOINT_CONFIDENCE: float = 0.3
    MIN_ARM_EXTENSION: float = 0.5
    
    # Alert settings
    ALERT_COOLDOWN: float = 1.0
    MIN_DETECTION_FRAMES: int = 1
    
    # Crowd detection
    CROWD_THRESHOLD: int = 5
    MIN_VELOCITY_FOR_CROWD: int = 150
    
    # Performance optimization
    SKIP_FRAMES: int = 0  # Process every Nth frame (0 = all frames)
    MAX_PEOPLE_TO_CHECK: int = 10  # Limit people checked per frame
    USE_HALF_PRECISION: bool = False  # FP16 for faster inference (requires GPU)
    
    # Debug mode
    DEBUG_MODE: bool = True

config = Config()

# =========================
# OPTIMIZED GEOMETRY
# =========================
class GeometryUtils:
    """Vectorized geometric calculations"""
    
    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle using optimized dot product"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0
        
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    @staticmethod
    def batch_distances(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Calculate distances between multiple point pairs efficiently"""
        return np.linalg.norm(points1 - points2, axis=1)
    
    @staticmethod
    def is_arm_reaching(shoulder: np.ndarray, elbow: np.ndarray, 
                       wrist: np.ndarray, target: np.ndarray) -> Tuple[bool, float, float]:
        """Optimized arm reaching detection"""
        # Arm vector
        arm_vec = wrist - shoulder
        to_target = target - wrist
        
        # Quick distance check first
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target > 300:  # Early exit if too far
            return False, 180.0, 0.0
        
        # Calculate angle
        angle = GeometryUtils.angle_between_vectors(arm_vec, to_target)
        
        # Calculate extension using cached norms
        upper = np.linalg.norm(elbow - shoulder)
        fore = np.linalg.norm(wrist - elbow)
        reach = np.linalg.norm(arm_vec)
        
        extension = reach / (upper + fore + 1e-6)
        
        is_reaching = (angle < 90) and (extension > config.MIN_ARM_EXTENSION)
        return is_reaching, angle, extension

# =========================
# OPTIMIZED VELOCITY TRACKER
# =========================
class VelocityTracker:
    """Memory-efficient velocity tracking"""
    def __init__(self, window_size=3):
        self.window = window_size
        self.data = {}  # Compact storage
        self.max_entries = 50  # Prevent unbounded growth
    
    def update(self, idx: int, left: np.ndarray, right: np.ndarray, t: float) -> Tuple[float, float]:
        """Update with bounds checking"""
        if idx not in self.data:
            if len(self.data) >= self.max_entries:
                # Remove oldest entry
                oldest = min(self.data.keys())
                del self.data[oldest]
            
            self.data[idx] = {
                'l': deque(maxlen=self.window),
                'r': deque(maxlen=self.window),
                't': deque(maxlen=self.window)
            }
        
        d = self.data[idx]
        d['l'].append(left)
        d['r'].append(right)
        d['t'].append(t)
        
        if len(d['t']) < 2:
            return 0.0, 0.0
        
        # Vectorized velocity calculation
        dt = np.diff(list(d['t']))
        valid = dt > 0
        
        if not np.any(valid):
            return 0.0, 0.0
        
        # Calculate velocities for valid time deltas
        vels = []
        l_pts = np.array(list(d['l']))
        r_pts = np.array(list(d['r']))
        
        for i in range(len(dt)):
            if valid[i]:
                l_vel = np.linalg.norm(l_pts[i+1] - l_pts[i]) / dt[i]
                r_vel = np.linalg.norm(r_pts[i+1] - r_pts[i]) / dt[i]
                vels.append(max(l_vel, r_vel))
        
        return max(vels) if vels else 0.0, np.mean(vels) if vels else 0.0

# =========================
# OPTIMIZED ACTION DETECTOR
# =========================
class ActionDetector:
    """Streamlined action detection with spatial indexing"""
    def __init__(self):
        self.alert_history = {}
        self.detection_buffer = {}
        self.last_cleanup = time.time()
    
    def cleanup_old_data(self, current_time: float):
        """Periodic cleanup to prevent memory leaks"""
        if current_time - self.last_cleanup < 5.0:
            return
        
        # Remove old alerts beyond cooldown
        cutoff = current_time - config.ALERT_COOLDOWN * 2
        self.alert_history = {k: v for k, v in self.alert_history.items() if v > cutoff}
        
        # Clean detection buffers
        self.detection_buffer = {k: v for k, v in self.detection_buffer.items() 
                                if len(v) > 0}
        
        self.last_cleanup = current_time
    
    @staticmethod
    def get_neck(kp: np.ndarray) -> np.ndarray:
        """Fast neck calculation"""
        return (kp[5] + kp[6]) * 0.4 + kp[0] * 0.2
    
    def check_arm_quality(self, confs: np.ndarray, wrist_idx: int) -> bool:
        """Quick confidence check"""
        if confs is None:
            return True
        
        indices = [5, 7, 9] if wrist_idx == 9 else [6, 8, 10]
        return all(confs[i] >= config.MIN_KEYPOINT_CONFIDENCE for i in indices if i < len(confs))
    
    def check_interaction(self, att_idx: int, vic_idx: int,
                         att_kp: np.ndarray, att_conf: np.ndarray,
                         vic_kp: np.ndarray, vic_box: np.ndarray,
                         vel: Tuple[float, float], n_people: int,
                         t: float) -> Tuple[bool, dict]:
        """Optimized interaction check"""
        
        # Quick spatial filter
        vic_neck = self.get_neck(vic_kp)
        bbox_size = max(vic_box[2] - vic_box[0], vic_box[3] - vic_box[1])
        prox_thresh = config.PROXIMITY_THRESH * bbox_size
        
        # Check both wrists at once
        wrists = np.array([att_kp[9], att_kp[10]])
        dists = np.linalg.norm(wrists - vic_neck, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist > prox_thresh:
            return False, {}
        
        # Select closer wrist
        wrist_idx = 9 if min_idx == 0 else 10
        wrist = wrists[min_idx]
        hand = "LEFT" if min_idx == 0 else "RIGHT"
        
        # Quality check
        if not self.check_arm_quality(att_conf, wrist_idx):
            return False, {}
        
        # Arm geometry
        shoulder = att_kp[5 if wrist_idx == 9 else 6]
        elbow = att_kp[7 if wrist_idx == 9 else 8]
        
        is_reach, angle, ext = GeometryUtils.is_arm_reaching(shoulder, elbow, wrist, vic_neck)
        
        if not is_reach:
            return False, {}
        
        # Velocity check with adaptive threshold
        max_vel, _ = vel
        vel_thresh = config.MIN_VELOCITY_FOR_CROWD if n_people >= config.CROWD_THRESHOLD else config.VEL_THRESH
        
        very_close = min_dist < 40
        if not (max_vel > vel_thresh or very_close):
            return False, {}
        
        # Cooldown check
        pair_key = f"{att_idx}_{vic_idx}"
        if t - self.alert_history.get(pair_key, 0) <= config.ALERT_COOLDOWN:
            return False, {}
        
        # Stability buffer
        if pair_key not in self.detection_buffer:
            self.detection_buffer[pair_key] = deque(maxlen=config.MIN_DETECTION_FRAMES)
        
        self.detection_buffer[pair_key].append(True)
        
        if len(self.detection_buffer[pair_key]) < config.MIN_DETECTION_FRAMES:
            return False, {}
        
        # Trigger alert
        self.alert_history[pair_key] = t
        
        return True, {
            'attacker_idx': att_idx,
            'victim_idx': vic_idx,
            'neck': vic_neck,
            'wrist': wrist,
            'velocity': max_vel,
            'distance': min_dist,
            'hand': hand,
            'angle': angle,
            'extension': ext
        }
    
    def check_all_interactions(self, dets: List[dict], vels: Dict, t: float) -> List[dict]:
        """Optimized interaction checking with early exits"""
        self.cleanup_old_data(t)
        
        n_people = len(dets)
        if n_people < 2:
            return []
        
        # Limit people to check for performance
        max_check = min(config.MAX_PEOPLE_TO_CHECK, n_people)
        
        # Sort by velocity, check top movers
        sorted_idx = sorted(range(n_people), 
                          key=lambda x: vels.get(x, (0, 0))[0], 
                          reverse=True)[:max_check]
        
        alerts = []
        for att_idx in sorted_idx:
            # Skip if velocity too low
            if vels.get(att_idx, (0, 0))[0] < config.VEL_THRESH * 0.5:
                continue
            
            for vic_idx in range(n_people):
                if vic_idx == att_idx:
                    continue
                
                is_threat, info = self.check_interaction(
                    att_idx, vic_idx,
                    dets[att_idx]['keypoints'],
                    dets[att_idx]['confidences'],
                    dets[vic_idx]['keypoints'],
                    dets[vic_idx]['box'],
                    vels.get(att_idx, (0.0, 0.0)),
                    n_people,
                    t
                )
                
                if is_threat:
                    alerts.append(info)
                    if config.DEBUG_MODE:
                        print(f"🚨 P{att_idx}({info['hand']}) → P{vic_idx} "
                              f"| D:{info['distance']:.0f} V:{info['velocity']:.0f}")
        
        return alerts

# =========================
# OPTIMIZED VISUALIZATION
# =========================
class Visualizer:
    """Efficient drawing with pre-computed colors"""
    
    SKELETON = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    @staticmethod
    def draw_skeleton(frame: np.ndarray, kp: np.ndarray, color=(0, 255, 0)):
        """Optimized skeleton drawing"""
        # Draw lines
        for s, e in Visualizer.SKELETON:
            if s < len(kp) and e < len(kp):
                p1, p2 = kp[s].astype(int), kp[e].astype(int)
                if not (p1[0] == 0 and p1[1] == 0) and not (p2[0] == 0 and p2[1] == 0):
                    cv2.line(frame, tuple(p1), tuple(p2), color, 2)
        
        # Draw keypoints with role-based colors
        for i, pt in enumerate(kp):
            if pt[0] == 0 and pt[1] == 0:
                continue
            
            p = tuple(pt.astype(int))
            if i in [9, 10]:  # Wrists
                cv2.circle(frame, p, 6, (255, 0, 255), -1)
            elif i in [0, 5, 6]:  # Head/shoulders
                cv2.circle(frame, p, 5, (0, 255, 255), -1)
            else:
                cv2.circle(frame, p, 4, color, -1)
    
    @staticmethod
    def draw_alerts(frame: np.ndarray, alerts: List[dict]):
        """Efficient alert visualization"""
        for a in alerts:
            neck, wrist = a['neck'].astype(int), a['wrist'].astype(int)
            
            cv2.line(frame, tuple(wrist), tuple(neck), (0, 0, 255), 4)
            cv2.circle(frame, tuple(neck), 35, (0, 0, 255), 4)
            cv2.circle(frame, tuple(wrist), 12, (0, 0, 255), -1)
            
            txt = f"THREAT! P{a['attacker_idx']}->P{a['victim_idx']}"
            cv2.putText(frame, txt, (neck[0] - 100, neck[1] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    @staticmethod
    def draw_hud(frame: np.ndarray, fc: int, fps: float, n_ppl: int, alerts: int):
        """Minimal HUD overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        texts = [
            f"Frame: {fc}",
            f"FPS: {fps:.1f}",
            f"People: {n_ppl}",
            f"Alerts: {alerts}"
        ]
        
        for i, txt in enumerate(texts):
            cv2.putText(frame, txt, (20, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# =========================
# OPTIMIZED MAIN SYSTEM
# =========================
class ThreatDetectionSystem:
    """High-performance threat detection"""
    
    def __init__(self):
        print("🚀 Initializing Optimized Threat Detection System...")
        
        # Load model
        self.model = YOLO(config.MODEL_PATH)
        
        # Enable FP16 only on CUDA with proper handling
        self.use_half = False
        if config.USE_HALF_PRECISION and torch.cuda.is_available():
            try:
                # Test if FP16 works
                self.model.to('cuda')
                self.use_half = True
                print("✓ Using FP16 precision on GPU")
            except Exception as e:
                print(f"⚠️  FP16 not available, using FP32: {e}")
                self.use_half = False
        
        self.vel_tracker = VelocityTracker(config.SMOOTHING_WINDOW)
        self.detector = ActionDetector()
        self.viz = Visualizer()
        self.total_alerts = 0
        
        print(f"✓ Model: {config.MODEL_PATH}")
        print(f"✓ Skip frames: {config.SKIP_FRAMES}")
        print(f"✓ Max people: {config.MAX_PEOPLE_TO_CHECK}")
    
    def process_frame(self, frame: np.ndarray, t: float) -> Tuple[np.ndarray, int]:
        """Optimized frame processing"""
        # Run inference with half precision if available
        kwargs = {
            'conf': config.CONF_THRES,
            'imgsz': config.IMG_SIZE,
            'verbose': False
        }
        
        if self.use_half:
            kwargs['half'] = True
        
        results = self.model(frame, **kwargs)
        
        if not results or results[0].keypoints is None:
            return frame, 0
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        kpts = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf.cpu().numpy() if hasattr(results[0].keypoints, 'conf') else None
        
        if len(boxes) == 0:
            return frame, 0
        
        # Build detections
        dets = [{'box': boxes[i], 'keypoints': kpts[i], 
                'confidences': confs[i] if confs is not None else None}
               for i in range(len(boxes))]
        
        # Calculate velocities
        vels = {}
        for i, d in enumerate(dets):
            kp = d['keypoints']
            vels[i] = self.vel_tracker.update(i, kp[9], kp[10], t)
        
        # Detect threats
        alerts = self.detector.check_all_interactions(dets, vels, t)
        self.total_alerts += len(alerts)
        
        # Visualize
        vis = frame.copy()
        for i, d in enumerate(dets):
            self.viz.draw_skeleton(vis, d['keypoints'])
        
        if alerts:
            self.viz.draw_alerts(vis, alerts)
        
        return vis, len(alerts)
    
    def run(self):
        """Main optimized loop"""
        cap = cv2.VideoCapture(config.SOURCE)
        
        if not cap.isOpened():
            print(f"❌ Cannot open: {config.SOURCE}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc, skipped = 0, 0
        start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing: {config.SOURCE}")
        print(f"{'='*60}\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            fc += 1
            time.sleep(0.001)  # Yield to OS
            
            # Frame skipping for performance
            if config.SKIP_FRAMES > 0 and fc % (config.SKIP_FRAMES + 1) != 0:
                skipped += 1
                continue
            
            t = fc / fps
            vis, n_alerts = self.process_frame(frame, t)
            
            # Stats
            elapsed = time.time() - start
            proc_fps = (fc - skipped) / elapsed if elapsed > 0 else 0
            
            self.viz.draw_hud(vis, fc, proc_fps, len(self.vel_tracker.data), self.total_alerts)
            
            cv2.imshow('Optimized Threat Detection', vis)
            
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"COMPLETE")
        print(f"{'='*60}")
        print(f"Frames: {fc} (Processed: {fc - skipped})")
        print(f"Alerts: {self.total_alerts}")
        print(f"Time: {elapsed:.2f}s")
        print(f"FPS: {(fc - skipped) / elapsed:.2f}")
        print(f"{'='*60}\n")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    try:
        system = ThreatDetectionSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()