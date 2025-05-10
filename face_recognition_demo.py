import face_recognition
import cv2
import numpy as np
import os

def load_and_display_image(image_path):
    """加载并显示图片"""
    # 使用face_recognition加载图片
    image = face_recognition.load_image_file(image_path)
    # 转换为OpenCV格式
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, image_cv

def detect_faces(image):
    """检测图片中的人脸"""
    # 检测人脸位置
    face_locations = face_recognition.face_locations(image)
    return face_locations

def get_face_landmarks(image, face_locations):
    """获取人脸特征点"""
    # 获取人脸特征点
    face_landmarks_list = face_recognition.face_landmarks(image, face_locations)
    return face_landmarks_list

def get_face_encodings(image, face_locations):
    """获取人脸编码"""
    # 获取人脸编码
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings

def compare_faces(known_face_encoding, face_encoding_to_check):
    """比较两个人脸编码"""
    # 比较人脸编码
    results = face_recognition.compare_faces([known_face_encoding], face_encoding_to_check)
    return results[0]

def draw_face_info(image_cv, face_locations, face_landmarks_list):
    """在图片上绘制人脸信息"""
    # 绘制人脸框
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # 绘制特征点
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                cv2.circle(image_cv, point, 2, (0, 0, 255), -1)
    
    return image_cv

def main():
    # 创建images文件夹（如果不存在）
    if not os.path.exists('images'):
        os.makedirs('images')
        print("请将测试图片放入images文件夹中")
        return

    # 示例：处理单张图片
    image_path = 'images/test.jpg'  # 请确保此路径下有图片
    if not os.path.exists(image_path):
        print(f"请将测试图片重命名为test.jpg并放入images文件夹中")
        return

    # 加载图片
    image, image_cv = load_and_display_image(image_path)
    
    # 检测人脸
    face_locations = detect_faces(image)
    print(f"检测到 {len(face_locations)} 个人脸")

    # 获取人脸特征点
    face_landmarks_list = get_face_landmarks(image, face_locations)
    
    # 获取人脸编码
    face_encodings = get_face_encodings(image, face_locations)
    
    # 在图片上绘制人脸信息
    result_image = draw_face_info(image_cv, face_locations, face_landmarks_list)
    
    # 显示结果
    cv2.imshow('Face Recognition Demo', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 