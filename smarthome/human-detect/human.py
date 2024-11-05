from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
import os
from datetime import datetime
from deepface import DeepFace
from retinaface import RetinaFace
import io

app = FastAPI()

# 创建输出目录
OUTPUT_DIR = "/Users/luolei/Desktop"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 定义颜色和边框设置
OUTER_COLOR = (255, 255, 255)  # 外层边框使用白色
INNER_COLOR = (0, 165, 255)    # 内层边框使用橙色
OUTER_THICKNESS = 6            # 外层边框粗细
INNER_THICKNESS = 3            # 内层边框粗细


def draw_detections(img, faces):
    """在图片上绘制检测结果"""
    img_result = img.copy()

    for face in faces:
        # 获取人脸框坐标
        facial_area = face['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']

        # 先画外层白色边框
        cv2.rectangle(img_result,
                      (x, y),
                      (x + w, y + h),
                      OUTER_COLOR,
                      OUTER_THICKNESS)

        # 再画内层橙色边框
        cv2.rectangle(img_result,
                      (x, y),
                      (x + w, y + h),
                      INNER_COLOR,
                      INNER_THICKNESS)

        # 如果有置信度分数，显示它
        if 'score' in face:
            confidence = face['score']
            # 添加置信度文本
            text = f"{confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_thickness = 2

            # 获取文本大小
            (text_width, text_height), baseline = cv2.getTextSize(text,
                                                                  font,
                                                                  font_scale,
                                                                  text_thickness)

            # 绘制文本背景（使用同样的双层效果）
            # 外层白色背景
            cv2.rectangle(img_result,
                          (x - 2, y - text_height - 8),
                          (x + text_width + 8, y + 2),
                          OUTER_COLOR,
                          -1)
            # 内层橙色背景
            cv2.rectangle(img_result,
                          (x, y - text_height - 6),
                          (x + text_width + 6, y),
                          INNER_COLOR,
                          -1)

            # 绘制文本
            cv2.putText(img_result,
                        text,
                        (x + 3, y - 7),
                        font,
                        font_scale,
                        (255, 255, 255),  # 白色文字
                        text_thickness)

    return img_result


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
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
        img_result = draw_detections(img, face_list)

        # 保存结果图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"result_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        success = cv2.imwrite(output_path, img_result)
        if not success:
            raise Exception("Failed to save the image")

        print(f"Successfully saved result to: {output_path}")

        return {
            "status": "success",
            "faces_detected": len(face_list),
            "faces": [
                {
                    "position": face['facial_area'],
                    "confidence": face.get('score', None)
                }
                for face in face_list
            ],
            "output_file": output_path
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.post("/detect_and_return/")
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
        img_result = draw_detections(img, face_list)

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
    uvicorn.run(app, host="0.0.0.0", port=8008)
