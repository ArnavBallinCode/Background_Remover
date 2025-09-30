# 🎭 AI Background Removal Tool

A powerful, easy-to-use background removal tool powered by the state-of-the-art BiRefNet (Bilateral Reference Network) model. Remove backgrounds from images with professional quality results using AI technology.

![Background Removal Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)

## ✨ Features

- **🤖 State-of-the-Art AI**: Uses BiRefNet model for high-quality background removal
- **🚀 GPU Acceleration**: Supports CUDA (NVIDIA) and MPS (Apple Silicon) for fast processing
- **� Multiple Input Methods**: 
  - Upload images directly
  - Process images from URLs
  - Batch file processing with download
- **🖼️ Interactive Interface**: Before/after comparison sliders
- **💾 Multiple Output Formats**: Transparent PNG files ready for use
- **� Web-Based**: No installation required for end users
- **⚡ Fast Processing**: Optimized for real-time background removal

## 🔧 Technical Specifications

- **Model**: ZhengPeng7/BiRefNet from Hugging Face
- **Framework**: PyTorch with Transformers
- **UI**: Gradio with custom components
- **Input Formats**: JPEG, PNG, WebP, and more
- **Output Format**: PNG with alpha channel (transparency)
- **GPU Support**: CUDA, MPS (Apple Silicon), CPU fallback

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Apple Silicon Mac (optional, for MPS acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArnavBallinCode/Background_Remover.git
   cd Background_Remover
   ```

2. **Create a conda environment (recommended)**
   ```bash
   conda create -n background-removal python=3.9
   conda activate background-removal
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:7860`
   - Or manually navigate to the provided URL

## 📖 Usage Guide

### Method 1: Image Upload
1. Go to the "🖼️ Image Upload" tab
2. Upload your image using the file picker
3. Use the slider to compare before and after results
4. Right-click on the result image to save

### Method 2: URL Processing
1. Go to the "🔗 URL Input" tab
2. Paste the URL of an image
3. View results with the interactive slider
4. Save the processed image

### Method 3: File Download
1. Go to the "📁 File Download" tab
2. Upload your image file
3. Download the processed PNG file directly

## 🏗️ Project Structure

```
background-removal/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── butterfly.jpg         # Example image
└── .gitignore           # Git ignore rules
```

## 🧠 How It Works

1. **Image Input**: The application accepts images in various formats
2. **Preprocessing**: Images are resized to 1024x1024 and normalized using ImageNet statistics
3. **AI Processing**: The BiRefNet model generates a segmentation mask
4. **Background Removal**: The mask is applied as an alpha channel for transparency
5. **Output**: Clean, transparent PNG images ready for use

## ⚙️ Configuration

### GPU Settings
The application automatically detects and uses the best available device:
- **Apple Silicon**: Uses MPS (Metal Performance Shaders)
- **NVIDIA GPUs**: Uses CUDA
- **CPU Fallback**: Works on any system

### Model Settings
- **Precision**: Optimized for high precision matrix operations
- **Memory**: Efficient memory usage with gradient-free inference
- **Performance**: Batch processing support for multiple images

## 🔧 Development

### Setting up Development Environment

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Background_Remover.git
   ```
3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Test thoroughly**
6. **Submit a pull request**

### Code Structure

- `app.py`: Main application with UI and processing logic
- `fn()`: Main processing function for UI integration
- `process()`: Core background removal logic
- `process_file()`: File-based processing with save functionality

## 📊 Performance

- **Processing Time**: ~2-5 seconds per image (GPU)
- **Memory Usage**: ~2-4GB VRAM (GPU mode)
- **Supported Sizes**: Up to 4K images
- **Quality**: Professional-grade results suitable for commercial use



### Areas for Contribution
- Performance optimizations
- Additional output formats
- Batch processing improvements
- UI/UX enhancements
- Documentation improvements


## 🙏 Acknowledgments

- **BiRefNet**: Thanks to the BiRefNet team for the excellent segmentation model
- **Hugging Face**: For providing the model hosting and transformers library
- **Gradio**: For the fantastic web UI framework
- **PyTorch**: For the deep learning framework

## 🔗 Links

- **Repository**: https://github.com/ArnavBallinCode/Background_Remover
- **BiRefNet Model**: https://huggingface.co/ZhengPeng7/BiRefNet
- **Gradio Documentation**: https://gradio.app/docs/

---

**Made with ❤️ by Arnav**
