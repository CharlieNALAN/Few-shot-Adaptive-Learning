import cv2
import numpy as np
import os

# 定义颜色阈值范围
lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])


# 图像预处理和颜色分割的函数
def process_image(image_path):
    try:
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Image at {image_path} could not be opened or does not exist.")
            return

        # 高斯模糊去噪
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # 色彩空间转换
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # 阈值分割
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # 保存处理后的图像
        output_dir = os.path.join(os.path.dirname(image_path), 'processed')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存HSV图像
        output_hsv_path = os.path.join(output_dir, os.path.basename(image_path).replace('.bmp', '_hsv.bmp'))
        cv2.imwrite(output_hsv_path, hsv_image)



    except Exception as e:
        print(f"An error occurred: {e}")


# 数据集目录
dataset_directory = '../Trimmed'


# 遍历数据集中的所有文件
def process_dataset(dataset_directory):
    for root, dirs, files in os.walk(dataset_directory):
        for filename in files:
            if filename.endswith('.bmp'):
                image_path = os.path.join(root, filename)
                process_image(image_path)
                print(f'Processed {filename}')

process_dataset(dataset_directory)
