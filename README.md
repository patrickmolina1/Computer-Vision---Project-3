# Semantic Segmentation with U-Net and ResNet-34

A deep learning project implementing semantic segmentation using a custom U-Net architecture with ResNet-34 encoder for pixel-level image classification on the Pascal VOC dataset.

## 📋 Overview

This project implements a semantic segmentation model that can classify each pixel in an image into one of 21 classes from the Pascal VOC dataset. The model uses a U-Net architecture with a pre-trained ResNet-34 encoder and includes various loss functions and optimization strategies.

## 🎯 Features

- **Custom U-Net Architecture**: Modified U-Net with ResNet-34 encoder for improved feature extraction
- **Multiple Loss Functions**: Cross-entropy, Dice Loss, and Focal Loss implementations
- **Data Augmentation**: Comprehensive augmentation pipeline using Albumentations
- **Interactive GUI**: User-friendly interface for model inference on custom images
- **TensorBoard Integration**: Real-time training monitoring and visualization
- **Multiple Optimizers**: Support for Adam, SGD, and other optimization algorithms

## 🏗️ Architecture

The model consists of:
- **Encoder**: Pre-trained ResNet-34 backbone for feature extraction
- **Decoder**: U-Net style decoder with skip connections and attention gates
- **Output**: 21-class semantic segmentation (Pascal VOC classes)

### Pascal VOC Classes
Background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Pascal VOC 2012 dataset

### Installation

1. Clone the repository:
```bash
git clone https://github.com/patrickmolina1/Computer-Vision---Project-3.git
cd Computer-Vision---Project-3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Pascal VOC 2012 dataset and place it in the `data/` directory

### Usage

#### Training the Model
Open and run the `semantic_segmentation.ipynb` notebook to:
- Load and preprocess the Pascal VOC dataset
- Train the model with different loss functions
- Monitor training progress with TensorBoard
- Save trained models

#### Using the GUI
Launch the interactive GUI for inference:
```bash
jupyter notebook gui.ipynb
```

The GUI allows you to:
- Select from pre-trained models
- Upload custom images
- Visualize segmentation results
- Compare different model performances

## 📊 Model Performance

The project includes trained models with different loss functions:
- `Adam_CrossEntropy.pth`: Standard cross-entropy loss
- `Adam_DiceLoss.pth`: Dice loss for better handling of class imbalance
- `Adam_FocalLoss.pth`: Focal loss for hard example mining

## 🔧 Technical Details

### Loss Functions
- **Cross-Entropy Loss**: Standard multi-class classification loss
- **Dice Loss**: Optimizes for IoU-like metrics, good for imbalanced datasets
- **Focal Loss**: Focuses on hard-to-classify pixels

### Data Augmentation
- Random horizontal flips
- Random rotations
- Color jittering
- Gaussian blur
- Normalization

### Evaluation Metrics
- Mean Intersection over Union (mIoU)
- Pixel accuracy
- Per-class IoU scores

## 📁 Project Structure

```
├── semantic_segmentation.ipynb  # Main training notebook
├── gui.ipynb                   # Interactive GUI for inference
├── report.ipynb               # Analysis and results
├── utils.py                   # Model architecture and utilities
├── requirements.txt           # Project dependencies
├── models/                    # Trained model weights
│   ├── Adam_CrossEntropy.pth
│   ├── Adam_DiceLoss.pth
│   └── Adam_FocalLoss.pth
└── tensor_board/             # TensorBoard logs
```

## 🛠️ Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
Pillow>=8.3.0
matplotlib>=3.4.0
albumentations>=1.0.0
tensorboard>=2.6.0
opencv-python>=4.5.0
jupyter>=1.0.0
tqdm>=4.62.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

## 📈 Results

The model achieves competitive performance on the Pascal VOC validation set:
- Mean IoU: ~65-70% (varies by loss function)
- Pixel Accuracy: ~85-90%
- Training monitored via TensorBoard for loss curves and metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Pascal VOC dataset creators
- PyTorch team for the deep learning framework
- ResNet architecture from "Deep Residual Learning for Image Recognition"
- U-Net architecture from "U-Net: Convolutional Networks for Biomedical Image Segmentation"

## 📧 Contact

Patrick Molina - [GitHub](https://github.com/patrickmolina1)

Project Link: [https://github.com/patrickmolina1/Computer-Vision---Project-3](https://github.com/patrickmolina1/Computer-Vision---Project-3)