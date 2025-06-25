import argparse
import sys
import cv2
import numpy as np
from PIL import Image
import os
import glob

class InteractiveCorrection:
    def __init__(self):
        self.correction_made = False
        self.new_roi = None
        self.mouse_start = None
        self.mouse_end = None
        self.drawing = False
        self.current_img = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_start = (x, y)
            self.drawing = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.mouse_end = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_end = (x, y)
            self.drawing = False
            if self.mouse_start and self.mouse_end:
                x1, y1 = self.mouse_start
                x2, y2 = self.mouse_end
                # Ensure we have a valid rectangle
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.new_roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                    self.correction_made = True
    
    def get_user_correction(self, img, frame_number, detection_box=None):
        """
        Show the image to user and allow them to correct the subject detection
        Returns: (corrected, new_roi) where corrected is bool and new_roi is (x,y,w,h) or None
        """
        self.correction_made = False
        self.new_roi = None
        self.current_img = img.copy()
        
        # Create window
        window_name = f"Frame {frame_number} - Correction"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Resize for display if needed
        display_img = img.copy()
        h, w = display_img.shape[:2]
        scale = 1.0
        if h > 800 or w > 800:
            scale = min(800/h, 800/w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        while True:
            temp_img = display_img.copy()
            
            # Draw current detection box if provided
            if detection_box:
                x, y, w, h = detection_box
                # Scale the box for display
                scaled_box = (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
                cv2.rectangle(temp_img, (scaled_box[0], scaled_box[1]), 
                            (scaled_box[0] + scaled_box[2], scaled_box[1] + scaled_box[3]), 
                            (0, 0, 255), 2)  # Red for current detection
                cv2.putText(temp_img, "Current Detection", (scaled_box[0], scaled_box[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw user's selection while dragging
            if self.drawing and self.mouse_start and self.mouse_end:
                cv2.rectangle(temp_img, self.mouse_start, self.mouse_end, (0, 255, 0), 2)
            
            # Draw completed selection
            if self.new_roi:
                x, y, w, h = self.new_roi
                # Scale back from display coordinates to original coordinates
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                orig_w = int(w / scale)
                orig_h = int(h / scale)
                
                # Draw on display image (scaled coordinates)
                cv2.rectangle(temp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(temp_img, "New Selection", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(temp_img, "Drag to select new subject area", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(temp_img, "Press ENTER to accept, SPACE to skip, ESC to exit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, temp_img)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 13:  # ENTER key
                cv2.destroyWindow(window_name)
                if self.new_roi:
                    # Convert back to original image coordinates
                    x, y, w, h = self.new_roi
                    orig_roi = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                    return True, orig_roi
                else:
                    return False, None
                    
            elif key == 32:  # SPACE key (skip)
                cv2.destroyWindow(window_name)
                return False, None
                
            elif key == 27:  # ESC key (exit)
                cv2.destroyWindow(window_name)
                return None, None  # Signal to exit program
        
        return False, None

def create_detection_boxes_gif(original_images, detection_info, output_path):
    """
    Create a GIF showing the detection boxes on the original images
    """
    gif_images = []
    
    for i, (img, det_info) in enumerate(zip(original_images, detection_info)):
        # Create a copy of the original image
        display_img = img.copy()
        
        # Draw detection box
        box_x, box_y, box_w, box_h = det_info['box']
        
        # Choose color based on detection method
        if det_info['method'] == 'reference':
            color = (0, 255, 0)  # Green for reference
        elif det_info['method'] == 'user_corrected':
            color = (255, 0, 255)  # Magenta for user corrected
        elif det_info['confidence'] > 0.6:
            color = (0, 255, 0)  # Green for high confidence
        elif det_info['confidence'] > 0.3:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence/fallback
        
        # Draw rectangle with thicker line for visibility
        cv2.rectangle(display_img, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 3)
        
        # Add frame number and detection info
        cv2.putText(display_img, f"Frame {i+1}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_img, f"{det_info['method']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert BGR to RGB for PIL
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        gif_images.append(Image.fromarray(rgb_img))
    
    # Save as GIF
    gif_images[0].save(
        output_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=500,
        loop=0
    )
    
    print(f"Detection boxes GIF created: {output_path}")

def create_cropped_gif(aligned_images, output_path):
    """
    Create a GIF with black space cropped out while keeping subject centered
    """
    # Find the common non-black region across all images
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    
    for img_array in aligned_images:
        # Convert to grayscale to find non-black regions
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find all non-black pixels (threshold slightly above 0 to handle compression artifacts)
        non_black_coords = np.where(gray > 0)
        
        if len(non_black_coords[0]) > 0:
            y_coords, x_coords = non_black_coords
            min_x = min(min_x, np.min(x_coords))
            min_y = min(min_y, np.min(y_coords))
            max_x = max(max_x, np.max(x_coords))
            max_y = max(max_y, np.max(y_coords))
    
    # Convert to integers and add some padding
    min_x, min_y = int(min_x), int(min_y)
    max_x, max_y = int(max_x), int(max_y)
    
    # Calculate crop dimensions
    crop_width = max_x - min_x + 1
    crop_height = max_y - min_y + 1
    
    # Ensure we have valid crop dimensions
    if crop_width <= 0 or crop_height <= 0:
        print("Warning: Could not determine crop region, using original images")
        cropped_images = [Image.fromarray(img) for img in aligned_images]
    else:
        # Crop all images to the same region
        cropped_images = []
        for img_array in aligned_images:
            cropped = img_array[min_y:max_y+1, min_x:max_x+1]
            cropped_images.append(Image.fromarray(cropped))
        
        print(f"Cropped from original size to {crop_width}x{crop_height}")
    
    # Save as GIF
    cropped_images[0].save(
        output_path,
        save_all=True,
        append_images=cropped_images[1:],
        duration=500,
        loop=0
    )
    
    print(f"Cropped GIF created: {output_path}")

def robust_manual_alignment_with_correction(image_paths, output_gif_path):
    """
    Enhanced robust alignment with user correction capability
    """
    # Load all images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    if not images:
        print("No valid images found!")
        return
    
    # Display first image and let user select the subject
    print("Select the subject in the first image by drawing a rectangle around it.")
    print("Press SPACE or ENTER to confirm selection, ESC to exit")
    
    first_img = images[0].copy()
    
    # Let user select ROI (Region of Interest)
    roi = cv2.selectROI("Select Subject - Press SPACE when done", first_img, False, False)
    cv2.destroyAllWindows()
    
    if roi == (0, 0, 0, 0):
        print("No selection made. Exiting...")
        return
    
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("Invalid selection. Exiting...")
        return
        
    print(f"Selected region: x={x}, y={y}, width={w}, height={h}")
    
    # Extract the template (selected subject) from first image
    template = first_img[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Calculate center of selected region in first image
    template_center_x = x + w // 2
    template_center_y = y + h // 2
    
    # Get dimensions for final images
    img_h, img_w = first_img.shape[:2]
    target_center_x, target_center_y = img_w // 2, img_h // 2
    
    aligned_images = []
    detection_info = []
    
    # Initialize ORB detector for feature-based matching
    orb = cv2.ORB_create(nfeatures=1000)
    template_keypoints, template_descriptors = orb.detectAndCompute(template_gray, None)
    
    # Create SIFT detector for better feature detection (if available)
    try:
        sift = cv2.SIFT_create()
        template_kp_sift, template_desc_sift = sift.detectAndCompute(template_gray, None)
        use_sift = True
    except:
        use_sift = False
        print("SIFT not available, using ORB for feature matching")
    
    # Initialize correction system
    corrector = InteractiveCorrection()
    
    # Process each image
    for i, img in enumerate(images):
        current_img = img.copy()
        
        if i == 0:
            # For first image, calculate offset to center the selected subject
            offset_x = target_center_x - template_center_x
            offset_y = target_center_y - template_center_y
            detection_info.append({'box': (x, y, w, h), 'method': 'reference', 'confidence': 1.0})
            print(f"Image 1: Reference image")
        else:
            img_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Multi-scale template matching
            best_match = multi_scale_template_matching(img_gray, template_gray, scales=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
            
            found_center_x = None
            found_center_y = None
            detection_method = "unknown"
            confidence = 0.0
            detection_box = None
            
            if best_match['confidence'] > 0.4:  # Good template match
                found_center_x = best_match['center'][0]
                found_center_y = best_match['center'][1]
                det_w = int(w * best_match['scale'])
                det_h = int(h * best_match['scale'])
                det_x = found_center_x - det_w // 2
                det_y = found_center_y - det_h // 2
                detection_box = (det_x, det_y, det_w, det_h)
                detection_method = 'template'
                confidence = best_match['confidence']
                print(f"Image {i+1}: Template matching (scale={best_match['scale']:.2f}, confidence={best_match['confidence']:.3f})")
            else:
                # Try feature-based matching as fallback
                print(f"Image {i+1}: Template matching failed (confidence={best_match['confidence']:.3f}), trying feature matching...")
                
                feature_result = feature_based_matching(img_gray, template_gray, template_kp_sift if use_sift else template_keypoints, 
                                                      template_desc_sift if use_sift else template_descriptors, use_sift)
                
                if feature_result['success']:
                    found_center_x = feature_result['center'][0]
                    found_center_y = feature_result['center'][1]
                    det_x = found_center_x - w // 2
                    det_y = found_center_y - h // 2
                    detection_box = (det_x, det_y, w, h)
                    detection_method = 'feature'
                    confidence = feature_result['matches']/100.0
                    print(f"  Feature matching succeeded (matches={feature_result['matches']})")
                else:
                    # Use fallback
                    print("  All methods failed, using image center as fallback")
                    found_center_x = img_w // 2
                    found_center_y = img_h // 2
                    det_x = found_center_x - w // 2
                    det_y = found_center_y - h // 2
                    detection_box = (det_x, det_y, w, h)
                    detection_method = 'fallback'
                    confidence = 0.0
            
            # Store detection info
            detection_info.append({'box': detection_box, 'method': detection_method, 'confidence': confidence})
            
            # Check if we should offer correction (low confidence or user wants to check)
            should_offer_correction = (confidence < 0.93) or (detection_method == 'fallback')
            
            if should_offer_correction:
                print(f"  Low confidence detection. Offering correction opportunity...")
                corrected, new_roi = corrector.get_user_correction(current_img, i+1, detection_box)
                
                if corrected is None:  # User pressed ESC to exit
                    print("User cancelled. Exiting...")
                    return
                elif corrected and new_roi:  # User made a correction
                    x_new, y_new, w_new, h_new = new_roi
                    print(f"  User corrected subject to: x={x_new}, y={y_new}, width={w_new}, height={h_new}")
                    
                    # Update template with the corrected region
                    new_template = current_img[y_new:y_new+h_new, x_new:x_new+w_new]
                    template_gray = cv2.cvtColor(new_template, cv2.COLOR_BGR2GRAY)
                    w, h = w_new, h_new  # Update template dimensions
                    
                    # Update feature descriptors for the new template
                    template_keypoints, template_descriptors = orb.detectAndCompute(template_gray, None)
                    if use_sift:
                        template_kp_sift, template_desc_sift = sift.detectAndCompute(template_gray, None)
                    
                    # Use the corrected center
                    found_center_x = x_new + w_new // 2
                    found_center_y = y_new + h_new // 2
                    
                    # Update detection info
                    detection_info[-1] = {'box': (x_new, y_new, w_new, h_new), 'method': 'user_corrected', 'confidence': 1.0}
                    print(f"  Template updated with user correction")
            
            # Calculate offset needed to center the subject
            offset_x = target_center_x - found_center_x
            offset_y = target_center_y - found_center_y
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        
        # Apply translation
        aligned_img = cv2.warpAffine(current_img, translation_matrix, (img_w, img_h))
        
        # Convert BGR to RGB for PIL
        aligned_images.append(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    
    # Convert to PIL Images and create GIF
    pil_images = [Image.fromarray(img) for img in aligned_images]
    
    # Save as GIF
    pil_images[0].save(
        output_gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=500,
        loop=0
    )
    
    print(f"Enhanced GIF created successfully: {output_gif_path}")
    
    # Create additional GIFs as requested
    # 1. GIF showing detection boxes on original images
    detection_gif_path = output_gif_path.replace('.gif', '_detection_boxes.gif')
    create_detection_boxes_gif(images, detection_info, detection_gif_path)
    
    # 2. GIF with black space cropped out
    cropped_gif_path = output_gif_path.replace('.gif', '_cropped.gif')
    create_cropped_gif(aligned_images, cropped_gif_path)
    
    # Show final preview
    preview_alignment_with_detection(images, aligned_images, detection_info, "Final Alignment Preview")

def multi_scale_template_matching(img, template, scales):
    """
    Perform template matching at multiple scales to handle size changes
    """
    best_match = {'confidence': 0, 'location': (0, 0), 'scale': 1.0, 'center': (0, 0)}
    template_h, template_w = template.shape
    
    for scale in scales:
        # Resize template
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)
        
        if scaled_w < 10 or scaled_h < 10 or scaled_w > img.shape[1] or scaled_h > img.shape[0]:
            continue
            
        scaled_template = cv2.resize(template, (scaled_w, scaled_h))
        
        # Try multiple matching methods
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        
        for method in methods:
            try:
                result = cv2.matchTemplate(img, scaled_template, method)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_match['confidence']:
                    best_match['confidence'] = max_val
                    best_match['location'] = max_loc
                    best_match['scale'] = scale
                    # Calculate center of matched region
                    center_x = max_loc[0] + scaled_w // 2
                    center_y = max_loc[1] + scaled_h // 2
                    best_match['center'] = (center_x, center_y)
            except:
                continue
    
    return best_match

def feature_based_matching(img, template, template_keypoints, template_descriptors, use_sift=True):
    """
    Use feature matching to find the subject even with rotation and significant changes
    """
    if template_descriptors is None or len(template_keypoints) < 10:
        return {'success': False, 'center': (0, 0), 'matches': 0}
    
    try:
        if use_sift:
            sift = cv2.SIFT_create()
            img_keypoints, img_descriptors = sift.detectAndCompute(img, None)
            # Use FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            orb = cv2.ORB_create(nfeatures=1000)
            img_keypoints, img_descriptors = orb.detectAndCompute(img, None)
            # Use BFMatcher for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        if img_descriptors is None or len(img_keypoints) < 10:
            return {'success': False, 'center': (0, 0), 'matches': 0}
        
        # Match features
        if use_sift:
            matches = flann.knnMatch(template_descriptors, img_descriptors, k=2)
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
        else:
            matches = bf.match(template_descriptors, img_descriptors)
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]  # Keep best 50 matches
        
        if len(good_matches) < 10:  # Need at least 10 good matches
            return {'success': False, 'center': (0, 0), 'matches': len(good_matches)}
        
        # Calculate center of matched points in the current image
        matched_points = []
        for match in good_matches:
            img_point = img_keypoints[match.trainIdx].pt
            matched_points.append(img_point)
        
        if matched_points:
            center_x = sum(p[0] for p in matched_points) / len(matched_points)
            center_y = sum(p[1] for p in matched_points) / len(matched_points)
            return {'success': True, 'center': (int(center_x), int(center_y)), 'matches': len(good_matches)}
        
    except Exception as e:
        print(f"    Feature matching error: {e}")
    
    return {'success': False, 'center': (0, 0), 'matches': 0}

def preview_alignment_with_detection(original_images, aligned_images, detection_info, window_name="Preview"):
    """
    Show a preview of the aligned images with detection boxes on original images
    """
    print("Showing alignment preview with detection boxes.")
    print("Press any key to advance to next frame, ESC to exit preview.")
    
    for i, (orig_img, aligned_img_array, det_info) in enumerate(zip(original_images, aligned_images, detection_info)):
        # Create display image from original
        display_img = orig_img.copy()
        
        # Draw detection box
        box_x, box_y, box_w, box_h = det_info['box']
        
        # Choose color based on detection method and confidence
        if det_info['method'] == 'reference':
            color = (0, 255, 0)  # Green for reference
        elif det_info['method'] == 'user_corrected':
            color = (255, 0, 255)  # Magenta for user corrected
        elif det_info['confidence'] > 0.6:
            color = (0, 255, 0)  # Green for high confidence
        elif det_info['confidence'] > 0.3:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence/fallback
        
        # Draw rectangle
        cv2.rectangle(display_img, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
        
        # Resize for display if image is too large
        h, w = display_img.shape[:2]
        if h > 800 or w > 800:
            scale = min(800/h, 800/w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        # Add frame number and detection info text
        cv2.putText(display_img, f"Frame {i+1}/{len(aligned_images)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_img, f"Method: {det_info['method']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_img, f"Confidence: {det_info['confidence']:.3f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Image Aligner")

    parser.add_argument("path", "path to images")

    args = parser.parse_args()
    
    # Get all image files from a directory
    image_paths = glob.glob(os.path.expanduser(args.path))
    # Filter for common image extensions
    image_extensions = ['.jpg', '.png']
    image_paths = [path for path in image_paths if any(path.lower().endswith(ext.lower()) for ext in image_extensions)]
    image_paths.sort()  # Sort to ensure consistent ordering
    
    if len(image_paths) < 2:
        print("Need at least 2 images to create a GIF")
    else:
        print("Enhanced alignment with user correction capability")
        robust_manual_alignment_with_correction(image_paths, "enhanced_aligned.gif")