# **License Plate Recognition System**

## **Overview**
This project presents a **License Plate Recognition (LPR)** system using **YOLO models** for object detection and **EasyOCR** for text recognition, designed to achieve real-time, accurate license plate detection. Trained on a diverse, hybrid dataset of over **52,000 labeled images**, the system demonstrates strong performance with high precision and recall, making it suitable for applications like traffic monitoring, automated tolling, and parking management.

## **Features**
- **YOLO-based Object Detection**: Uses **YOLO models** for detecting license plates in real-time.
- **EasyOCR for Text Recognition**: Extracts alphanumeric characters from detected license plates.
- **Hybrid Dataset**: Combines three different datasets containing a variety of challenging scenarios such as occlusions, lighting inconsistencies, and pose variations.
- **Data Augmentation**: Includes transformations like flipping, cropping, rotation, and color adjustments to simulate real-world conditions and improve model robustness.

## **Dataset**
The dataset consists of images sourced from multiple projects:
- **Dataset 1**: 21,175 images from a license plate recognition computer vision project.
- **Dataset 2**: 6,784 images from the license plates of vehicles in Turkey project.
- **Dataset 3**: 24,242 images from a vehicle registration plates project.

This hybrid dataset contains a total of **52,201 labeled images**, split into training, validation, and test sets:
- **Training**: 46,278 images (87%)
- **Validation**: 3,957 images (8%)
- **Test**: 1,966 images (4%)

### **Preprocessing**
- **Resize**: All images were resized to **640x640** pixels.
- **Data Augmentation**: Includes horizontal flipping, rotation, cropping, grayscale conversion, and adjustments in brightness, hue, saturation, and contrast.

## **Model Architecture**
We evaluated several **YOLO models**, including:
- **YOLOv8** and **YOLOv11** (with various configurations: **n**, **s**, **m**, **l**, **x**).
- **EasyOCR** for optical character recognition from detected license plates.

### **Key Insights**
- **YOLOv11x** achieved the highest mean Average Precision (**mAP**) of **0.98466 at IoU 50%** and **0.71605 at IoU 50-95%** after 20 epochs.
- **YOLOv11n**, a lightweight model, achieved high performance with similar precision and recall values after 100 epochs.
- The **YOLOv11m** model achieved the best balance of performance and accuracy, with the highest **mAP@50-95** of **0.71743**.

## **Performance Evaluation**
The system was evaluated using **Precision**, **Recall**, and **mean Average Precision (mAP)** at both **IoU 50%** and **IoU 50-95%**. The results showed improvements in recall and mAP with extended training, particularly in **YOLOv8** and **YOLOv11** models trained for **100 epochs**.

### **Example Results:**
- **YOLOv8** (10 epochs) achieved:
  - **Precision (P)**: 0.974
  - **Recall (R)**: 0.954
  - **mAP@50**: 0.978
  - **mAP@50-95**: 0.682

- **YOLOv11** (10 epochs) achieved:
  - **Precision (P)**: 0.981
  - **Recall (R)**: 0.951
  - **mAP@50**: 0.981
  - **mAP@50-95**: 0.682

Training for **100 epochs** significantly enhanced detection capabilities, improving **mAP@50-95** by up to **4.25%**.

## **Ablation Study**
An **Ablation Study** was conducted to evaluate the performance of different variants of **YOLOv11** models trained on the hybrid dataset. The study aimed to assess the trade-offs between model precision, recall, and computational efficiency.

### **YOLOv11 Model Variants**
The following **YOLOv11** variants were evaluated:
- **YOLOv11n** (lightweight)
- **YOLOv11s**
- **YOLOv11m**
- **YOLOv11l**
- **YOLOv11x** (largest model)

### **Performance Comparison**

The results of the ablation study are summarized in the table below:

| **Model**     | **Precision (B)** | **Recall (B)** | **mAP@50 (B)** | **mAP@50-95 (B)** |
|---------------|-------------------|----------------|----------------|-------------------|
| **YOLOv11x**  | 0.97384           | 0.95932        | 0.98387        | 0.71605           |
| **YOLOv11l**  | 0.96733           | 0.96591        | 0.98497        | 0.71454           |
| **YOLOv11m**  | 0.97035           | 0.96761        | 0.98582        | 0.71743           |
| **YOLOv11s**  | 0.97312           | 0.96170        | 0.98477        | 0.71344           |
| **YOLOv11n**  | 0.97125           | 0.95849        | 0.98235        | 0.71064           |

### **Key Findings**
- **YOLOv11m** achieved the highest **mAP@50-95** (**0.71743**), making it the best choice for applications requiring high accuracy.
- **YOLOv11x** achieved the highest **Precision (B)** (**0.97384**), suitable for applications prioritizing detection accuracy.
- **YOLOv11l** achieved the highest **Recall (B)** (**0.96591**), indicating its capability to minimize missed detections.
- **YOLOv11n** and **YOLOv11s** are lightweight models that balance computational efficiency with satisfactory detection metrics, making them ideal for deployment on edge devices.

These results provide insights into the trade-offs between precision, recall, and computational requirements for each variant, allowing tailored model selection based on application-specific needs.
