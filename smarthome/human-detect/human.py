from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
import os
from datetime import datetime
from deepface import DeepFace
from retinaface import RetinaFace
import io
from typing import Optional

app = FastAPI()

# 创建输出目录
OUTPUT_DIR = "/Users/luolei/Desktop"
# 服务器端口
SERVER_PORT = 8008

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 定义颜色和边框设置
OUTER_COLOR = (255, 255, 255)  # 外层边框使用白色
INNER_COLOR = (0, 165, 255)    # 内层边框使用橙色
OUTER_THICKNESS = 6            # 外层边框粗细
INNER_THICKNESS = 3            # 内层边框粗细


def draw_detections_with_info(img, faces):
    """在图片上绘制检测结果和额外信息"""
    img_result = img.copy()

    for face in faces:
        # 获取人脸区域
        facial_area = face['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']

        # 画双层边框
        cv2.rectangle(img_result,
                      (x, y),
                      (x + w, y + h),
                      OUTER_COLOR,
                      OUTER_THICKNESS)
        cv2.rectangle(img_result,
                      (x, y),
                      (x + w, y + h),
                      INNER_COLOR,
                      INNER_THICKNESS)

        # 准备显示的信息
        info_lines = []
        if face.get('score'):
            info_lines.append(f"Conf: {face['score']:.2f}")
        if face.get('age'):
            info_lines.append(f"Age: {face['age']}")
        if face.get('gender'):
            info_lines.append(f"Gender: {face['gender']}")
        if face.get('dominant_emotion'):
            info_lines.append(f"Emotion: {face['dominant_emotion']}")

        # 文本显示设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        line_height = 25
        padding = 5

        # 计算所有文本的总高度
        total_height = len(info_lines) * line_height

        # 确定文本起始位置（避免超出图像边界）
        start_y = max(total_height + padding, y)

        # 绘制信息
        for i, line in enumerate(info_lines):
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(
                line, font, font_scale, text_thickness)

            # 计算文本位置
            text_y = start_y - (i * line_height)

            # 确保文本框不会超出图像边界
            text_x = min(x, img_result.shape[1] - text_width - padding)

            # 绘制文本背景（双层效果）
            cv2.rectangle(img_result,
                          (text_x - padding, text_y - text_height - padding),
                          (text_x + text_width + padding, text_y + padding),
                          OUTER_COLOR,
                          -1)
            cv2.rectangle(img_result,
                          (text_x - padding + 2, text_y -
                           text_height - padding + 2),
                          (text_x + text_width + padding - 2, text_y + padding - 2),
                          INNER_COLOR,
                          -1)

            # 绘制文本
            cv2.putText(img_result,
                        line,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        text_thickness)

    return img_result


def convert_to_native_types(obj):
    """将 numpy 类型转换为 Python 原生类型"""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@app.post("/analyze")
async def detect(
    file: UploadFile = File(...),
    save_render: Optional[bool] = Query(
        default=False,
        description="Whether to save the rendered image with face detection markers"
    )
):
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image"}
            )

        # 使用 RetinaFace 进行人脸检测
        faces = RetinaFace.detect_faces(img)

        # 转换检测结果格式
        face_list = []
        if isinstance(faces, dict):
            for face_key in faces:
                face_data = faces[face_key]
                facial_area = {
                    'x': int(face_data['facial_area'][0]),
                    'y': int(face_data['facial_area'][1]),
                    'w': int(face_data['facial_area'][2] - face_data['facial_area'][0]),
                    'h': int(face_data['facial_area'][3] - face_data['facial_area'][1])
                }

                face_img = img[
                    facial_area['y']:facial_area['y']+facial_area['h'],
                    facial_area['x']:facial_area['x']+facial_area['w']
                ]

                try:
                    analysis = DeepFace.analyze(
                        face_img,
                        actions=['age', 'gender', 'race', 'emotion'],
                        enforce_detection=False
                    )

                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    gender = "Woman" if analysis['gender']['Woman'] > 50 else "Man"

                    face_info = {
                        'position': facial_area,
                        'confidence': float(face_data['score']) if 'score' in face_data else None,
                        'age': int(analysis['age']),
                        'gender': gender,
                        'dominant_race': str(analysis['dominant_race']),
                        'dominant_emotion': str(analysis['dominant_emotion']),
                        'emotion': {k: float(v) for k, v in analysis['emotion'].items()},
                        'race': {k: float(v) for k, v in analysis['race'].items()}
                    }

                except Exception as e:
                    print(f"Face analysis failed: {str(e)}")
                    face_info = {
                        'position': facial_area,
                        'confidence': float(face_data['score']) if 'score' in face_data else None
                    }

                face_list.append(face_info)

        output_path = None
        if save_render:
            # 准备绘制信息
            face_info_list = []
            for face in face_list:
                face_info = {
                    'facial_area': face['position'],
                    'score': face['confidence'],
                    'age': face.get('age'),
                    'gender': face.get('gender'),
                    'dominant_emotion': face.get('dominant_emotion')
                }
                face_info_list.append(face_info)

            # 绘制标记
            img_result = draw_detections_with_info(img, face_info_list)

            # 保存带标记的图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"result_{timestamp}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, img_result)

        # 准备响应数据
        response_data = {
            "status": "success",
            "faces_detected": len(face_list),
            "faces": convert_to_native_types(face_list),
        }

        # 只在保存了渲染图片时添加文件路径
        if output_path:
            response_data["output_file"] = output_path

        return response_data

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.post("/detect_and_return")
async def detect_and_return(file: UploadFile = File(...)):
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image"}
            )

        # 使用 RetinaFace 进行人脸检测
        faces = RetinaFace.detect_faces(img)

        # 转换检测结果格式
        face_list = []
        if isinstance(faces, dict):  # 检测到人脸
            for face_key in faces:
                face_data = faces[face_key]
                facial_area = {
                    'x': int(face_data['facial_area'][0]),
                    'y': int(face_data['facial_area'][1]),
                    'w': int(face_data['facial_area'][2] - face_data['facial_area'][0]),
                    'h': int(face_data['facial_area'][3] - face_data['facial_area'][1])
                }
                face_list.append({
                    'facial_area': facial_area,
                    'score': float(face_data['score']) if 'score' in face_data else None
                })

        # 在图片上绘制检测结果
        img_result = draw_detections_with_info(img, face_list)

        # 将图片编码为JPEG格式
        _, img_encoded = cv2.imencode('.jpg', img_result)

        # 返回处理后的图片
        return Response(
            content=img_encoded.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Faces-Detected": str(len(face_list)),
                "X-Detection-Status": "success" if face_list else "no_face"
            }
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
