SoniSight is an diagnostic web app that analyzes breast ultrasound images to identify normal vs. suspicious masses.  
It uses AI to highlight regions of interest and explain why a mass might appear abnormal.

This project was built for the Congressional App Challenge and inspired by my mother’s experience with uncertain biopsy results.

How It Works
1. Upload an ultrasound image or choose from built-in sample images.
2. The backend (FastAPI + OpenCV) extracts image features like:
   - Shape, margins, and texture
   - Contrast and circularity
3. Gemini AI analyzes those descriptors and outputs:
   - Probabilities for normal** vs suspicious
   - A short, human-readable rationale
4. A visual overlay highlights the detected region of interest.


Tech Stack
- Frontend: React + Vite (hosted on Vercel)
- **Site Link**: https://sonisight-pr.vercel.app   
- Backend: FastAPI + OpenCV (hosted on Render)  
- AI Model: Google Gemini 2.5 Flash  
- Dataset: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

