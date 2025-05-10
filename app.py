from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import os
import base64
from datetime import datetime

app = Flask(__name__)

# 确保上传文件夹存在
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_image(image_path):
    """处理图片并返回结果"""
    # 加载图片
    image = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 在图片上绘制人脸框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 保存处理后的图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(UPLOAD_FOLDER, f'processed_{timestamp}.jpg')
    cv2.imwrite(output_path, image)
    
    return {
        'face_count': len(faces),
        'processed_image': f'uploads/processed_{timestamp}.jpg'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file:
        # 保存上传的图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'original_{timestamp}.jpg'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 处理图片
        result = process_image(filepath)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 