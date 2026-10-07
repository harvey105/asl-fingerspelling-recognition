"""
src/preprocess_roi.py
Core localization engine for Hand ROI cropping and Alpha Mask extraction.
Implements MediaPipe Landmark detection with YCrCb + Otsu thresholding fallback.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import matplotlib.pyplot as plt
import random

# Ensure reproducible tests
random.seed(42)
np.random.seed(42)

class HandROIExtractor:
    def __init__(self, target_size=(200, 200), margin_pad=0.15):
        """
        Initializes the MediaPipe Hands model and extractor parameters.
        :param target_size: Tuple (W, H) for the final standardized crop.
        :param margin_pad: Float percentage (0.15 = 15%) to expand bounding box.
        """
        self.target_size = target_size
        self.margin_pad = margin_pad
        
        # Standard initialization (now works because protobuf is fixed)
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3  # Lowered slightly to catch difficult angles
        )

    def process_image(self, image_rgb):
        """
        Processes a single RGB image to extract the cropped hand and alpha mask.
        :param image_rgb: HxWxC numpy array (RGB format).
        :return: (cropped_rgb, alpha_mask_binary) both resized to target_size.
        """
        h, w, _ = image_rgb.shape
        results = self.hands_detector.process(image_rgb)
        
        bbox = None
        
        # 1. PRIMARY PASS: MediaPipe Landmark Bounding Box
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Apply 15% Margin Padding
            box_w, box_h = x_max - x_min, y_max - y_min
            pad_x = int(box_w * self.margin_pad)
            pad_y = int(box_h * self.margin_pad)
            
            bbox = (
                max(0, x_min - pad_x),
                max(0, y_min - pad_y),
                min(w, x_max + pad_x),
                min(h, y_max + pad_y)
            )
            
        # 2. FALLBACK PASS: YCrCb + Otsu thresholding if MediaPipe fails
        if bbox is None:
            bbox = self._fallback_skin_bbox(image_rgb)

        # 3. Crop the RGB image safely within bounds
        bx1, by1, bx2, by2 = bbox
        # Prevent zero-dimension crops if bounding box collapses
        if bx2 <= bx1 or by2 <= by1:
            bx1, by1, bx2, by2 = int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)
            
        cropped_rgb = image_rgb[by1:by2, bx1:bx2]
        
        # 4. Generate precise Alpha Mask strictly inside the cropped region
        alpha_mask = self._generate_alpha_mask(cropped_rgb)
        
        # 5. Standardize Resolution (Bilinear for RGB, Nearest for Binary Mask)
        cropped_standard = cv2.resize(cropped_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask_standard = cv2.resize(alpha_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is exactly 0 and 1
        mask_standard = (mask_standard > 0).astype(np.uint8)
        
        return cropped_standard, mask_standard

    def _fallback_skin_bbox(self, image_rgb):
        """Fallback method using YCrCb skin clustering and contour detection."""
        h, w, _ = image_rgb.shape
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        
        # Standard skin thresholding bounds in YCrCb space
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Find largest contour
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, box_w, box_h = cv2.boundingRect(largest_contour)
            
            # Apply padding
            pad_x = int(box_w * self.margin_pad)
            pad_y = int(box_h * self.margin_pad)
            return (
                max(0, x - pad_x),
                max(0, y - pad_y),
                min(w, x + box_w + pad_x),
                min(h, y + box_h + pad_y)
            )
        else:
            # Absolute worst-case scenario: center crop
            return (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))

    def _generate_alpha_mask(self, cropped_rgb):
        """Generates a crisp 1-bit binary mask from a tightly cropped hand image."""
        if cropped_rgb.size == 0:
            return np.ones((self.target_size[1], self.target_size[0]), dtype=np.uint8) * 255
            
        ycrcb = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2YCrCb)
        cr_channel = ycrcb[:, :, 1]
        
        _, mask = cv2.threshold(cr_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask


# =====================================================================
# LOCAL TESTING & VISUAL VALIDATION MODULE
# =====================================================================
if __name__ == "__main__":
    print("Initializing HandROIExtractor Test...")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw" / "asl_alphabet_train"
    out_figs = project_root / "docs" / "figures"
    out_figs.mkdir(parents=True, exist_ok=True)
    
    # Instantiate extractor
    extractor = HandROIExtractor(target_size=(200, 200), margin_pad=0.15)
    
    # Sample 5 distinct challenging classes
    test_classes = ["A", "M", "K", "nothing", "space"]
    
    fig, axes = plt.subplots(len(test_classes), 3, figsize=(10, 15))
    
    for row_idx, cls_name in enumerate(test_classes):
        cls_folder = data_dir / cls_name
        if not cls_folder.exists():
            print(f"Skipping {cls_name}, folder not found.")
            continue
            
        img_paths = list(cls_folder.glob("*.jpg"))
        test_img_path = random.choice(img_paths)
        
        # Load and convert to RGB
        img_bgr = cv2.imread(str(test_img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Run Pipeline
        cropped, mask = extractor.process_image(img_rgb)
        
        # Apply mask to crop to visualize the extracted foreground
        mask_255 = (mask * 255).astype(np.uint8)
        foreground_only = cv2.bitwise_and(cropped, cropped, mask=mask_255)
        
        # Plotting
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 0].set_title(f"Original ('{cls_name}')")
        axes[row_idx, 0].axis("off")
        
        axes[row_idx, 1].imshow(cropped)
        axes[row_idx, 1].set_title("Tight Crop (15% Pad)")
        axes[row_idx, 1].axis("off")
        
        axes[row_idx, 2].imshow(foreground_only)
        axes[row_idx, 2].set_title("Alpha Mask Extraction")
        axes[row_idx, 2].axis("off")

    plt.suptitle("Task 2.2: Hand ROI Cropping & Mask Pipeline Validation", fontsize=14, weight="bold")
    plt.tight_layout()
    
    save_path = out_figs / "roi_test_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n[SUCCESS] Pipeline Visual Validation saved to: {save_path.relative_to(project_root)}")
    plt.show()