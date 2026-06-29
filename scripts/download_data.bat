@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Brain Tumor Dataset Setup
echo ==========================================

REM Install required packages
pip install kagglehub pycocotools pillow numpy --upgrade

REM Create dataset folders
mkdir brain_tumor_unet\train\imgs
mkdir brain_tumor_unet\train\masks
mkdir brain_tumor_unet\valid\imgs
mkdir brain_tumor_unet\valid\masks
mkdir brain_tumor_unet\test\imgs
mkdir brain_tumor_unet\test\masks

REM Run Python conversion script
python dataset\coco_to_mask.py

echo.
echo Dataset conversion finished.
echo Output folder:
echo brain_tumor_unet\
echo.
pause
