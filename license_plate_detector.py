#!/usr/bin/env python
"""
License Plate Detection and Recognition Script

This script processes images in a given folder, detects license plates using a specified
YOLO model, and recognizes the text using various OCR engines. Results are saved in a 'results' folder.

Usage:
    python license_plate_detector.py --model <path_to_model> --input <input_folder> --ocr-method <method>

Example:
    python license_plate_detector.py --model runs/detect/train/weights/best.pt --input data/test_images --ocr-method easyocr
"""

import os
import sys
import argparse
import glob
import cv2
import time
import datetime
import numpy as np
from pathlib import Path
import torch

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Error: ultralytics module is not installed.")
    print("Install it with: pip install ultralytics")
    sys.exit(1)

# OCR Engine Initialization Functions
def init_easyocr_reader(gpu=True):
    """Initialize the easyOCR reader"""
    try:
        import easyocr
        print("Initializing easyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=gpu)
        return reader
    except ImportError:
        print("Error: easyocr module is not installed")
        print("Install it with: pip install easyocr")
        sys.exit(1)

def init_tesseract_ocr():
    """Initialize Tesseract OCR"""
    try:
        import pytesseract
        print("Initializing Tesseract OCR...")
        # Check if tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            return pytesseract
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract OCR is not installed on your system")
            print("Install it following instructions at: https://github.com/tesseract-ocr/tesseract")
            sys.exit(1)
    except ImportError:
        print("Error: pytesseract module is not installed")
        print("Install it with: pip install pytesseract")
        sys.exit(1)

def init_paddleocr(use_gpu=True):
    """Initialize PaddleOCR"""
    try:
        from paddleocr import PaddleOCR
        print("Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
        return ocr
    except ImportError:
        print("Error: paddleocr module is not installed")
        print("Install it with: pip install paddleocr")
        sys.exit(1)

def preprocess_plate_image(plate_img):
    """Preprocess the license plate image to improve OCR accuracy"""
    # Convert to grayscale
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply adaptive threshold as an alternative
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Return multiple versions for OCR to try
    return {"original": plate_img, "gray": gray, "binary": binary, "adaptive": adaptive}

def recognize_plate_text(plate_img, ocr_method, ocr_engine, confidence_threshold=0.3):
    """Recognize text in license plate using the specified OCR method"""
    # Preprocess the image for better OCR results
    processed_images = preprocess_plate_image(plate_img)
    
    results = []
    
    if ocr_method == "easyocr":
        # Try with each preprocessed image
        for img_type, img in processed_images.items():
            ocr_result = ocr_engine.readtext(img)
            for detection in ocr_result:
                if len(detection) >= 2:
                    if len(detection) >= 3 and detection[2] < confidence_threshold:
                        continue
                    text = detection[1]
                    results.append(text)
        
    elif ocr_method == "tesseract":
        # Configure tesseract for license plates (alphanumeric with limited special chars)
        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. '
        
        # Try with each preprocessed image
        for img_type, img in processed_images.items():
            if img_type in ["gray", "binary", "adaptive"]:  # Tesseract works with grayscale
                text = ocr_engine.image_to_string(img, config=config).strip()
                if text:
                    results.append(text)
    
    elif ocr_method == "paddleocr":
        # Try with original and grayscale only (PaddleOCR handles preprocessing internally)
        for img_type, img in {"original": processed_images["original"], 
                             "gray": processed_images["gray"]}.items():
            ocr_result = ocr_engine.ocr(img, cls=True)
            if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                for line in ocr_result[0]:
                    if len(line) >= 2 and line[1][1] >= confidence_threshold:
                        text = line[1][0]
                        results.append(text)
    
    # Remove duplicates while preserving order
    unique_results = []
    for item in results:
        if item not in unique_results:
            unique_results.append(item)
    
    return " ".join(unique_results)

def create_output_dir(input_folder):
    """Create output directory based on input folder name and timestamp"""
    # Get base folder name from input path
    base_folder = os.path.basename(os.path.normpath(input_folder))
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = os.path.join("results", f"{base_folder}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def process_images(model_path, input_folder, ocr_method="easyocr", device="0", confidence_threshold=0.25):
    """Process all images in the input folder and save results"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        return None
    
    # Create output directory
    output_dir = create_output_dir(input_folder)
    print(f"Results will be saved to: {output_dir}")
    
    # Create a text file for license plate texts
    plates_file_path = os.path.join(output_dir, "license_plates.txt")
    plates_file = open(plates_file_path, "w", encoding="utf-8")
    
    # Load YOLO model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Set model confidence threshold
    model.conf = confidence_threshold
    
    # Initialize OCR engine
    use_gpu = device != "cpu"
    if ocr_method == "easyocr":
        ocr_engine = init_easyocr_reader(gpu=use_gpu)
    elif ocr_method == "tesseract":
        ocr_engine = init_tesseract_ocr()
    elif ocr_method == "paddleocr":
        ocr_engine = init_paddleocr(use_gpu=use_gpu)
    else:
        print(f"Error: Unknown OCR method '{ocr_method}'")
        plates_file.close()
        return None
    
    # Get all images from input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        # Check for images in subdirectories
        image_files.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        plates_file.close()
        return None
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_path}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue
        
        # Get image filename without extension
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Detect license plates with YOLO
        results = model(img)
        
        # Create a copy of the image for drawing
        result_img = img.copy()
        
        # Check if any detections were made
        found_plates = False
        
        # Process each detection
        for r in results:
            boxes = r.boxes
            
            # If no detections, continue to next image
            if len(boxes) == 0:
                continue
                
            # Process each detected license plate
            for j, box in enumerate(boxes):
                # Get coordinates (convert from tensor to numpy if needed)
                if isinstance(box.xyxy, torch.Tensor):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                else:
                    x1, y1, x2, y2 = box.xyxy[0]
                
                # Get confidence score
                conf = float(box.conf[0])
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract the license plate region
                plate_img = img[y1:y2, x1:x2]
                
                # Skip if plate is too small
                if plate_img.shape[0] < 8 or plate_img.shape[1] < 8:
                    continue
                
                # Add confidence text
                cv2.putText(result_img, f"Conf: {conf:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save the cropped plate for debugging
                plate_path = os.path.join(output_dir, f"{filename}_plate_{j}.jpg")
                cv2.imwrite(plate_path, plate_img)
                
                # Recognize text using selected OCR method
                plate_text = recognize_plate_text(plate_img, ocr_method, ocr_engine)
                
                # Draw the recognized text at the top-left corner of the image
                if plate_text:
                    found_plates = True
                    print(f"Plate Text Found: {plate_text} (conf: {conf:.2f})")
                    
                    # Write the plate text to the file with filename reference
                    plates_file.write(f"{filename}: {plate_text} (conf: {conf:.2f})\n")

                    # Add black background for better visibility
                    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.rectangle(result_img, (10, 10), (10 + text_size[0], 40), (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(result_img, plate_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if not found_plates:
            print(f"No license plates detected in {img_path}")
            # Write to log file even when no plates are found
            plates_file.write(f"{filename}: No license plates detected\n")
        
        # Save the result image
        output_path = os.path.join(output_dir, f"{filename}_result.jpg")
        cv2.imwrite(output_path, result_img)
    
    print(f"Processing complete. Results saved to: {output_dir}")
    
    # Close the license plates file
    plates_file.close()
    print(f"License plate texts saved to: {plates_file_path}")
    
    return output_dir

def main():
    """Main function to parse arguments and run the script"""
    parser = argparse.ArgumentParser(description="License Plate Detection and Recognition")
    
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to YOLO model file (.pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input folder containing images")
    parser.add_argument("--ocr-method", type=str, default="easyocr",
                        choices=["easyocr", "tesseract", "paddleocr"],
                        help="OCR method to use: easyocr, tesseract, or paddleocr")
    parser.add_argument("--device", type=str, default="0",
                        help="Device for inference: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold for detection (0-1)")
    
    args = parser.parse_args()
    
    # Check if required modules are available
    if not YOLO_AVAILABLE:
        print("Error: ultralytics module is required")
        sys.exit(1)
    
    # Run the processing
    start_time = time.time()
    output_dir = process_images(
        args.model, 
        args.input, 
        ocr_method=args.ocr_method,
        device=args.device,
        confidence_threshold=args.confidence
    )
    elapsed_time = time.time() - start_time
    
    if output_dir:
        print(f"Processing completed in {elapsed_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
    else:
        print("Processing failed")

if __name__ == "__main__":
    main()
