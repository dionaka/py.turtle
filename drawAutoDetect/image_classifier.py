import sys # Import sys module
from PIL import Image
try:
    from transformers import pipeline
except ImportError:
    print("警告: 未检测到 'transformers' 库。图像分类功能将不可用。")
    print("您可以通过运行 'pip install transformers torch' 来安装它。")
    pipeline = None # Set pipeline to None if transformers is not available

def classify_image(image_path):
    """
    使用预训练的Hugging Face模型对图像进行分类。

    Args:
        image_path (str): 待分类图片的路径。

    Returns:
        list: 包含分类结果的列表，每个结果是一个字典，包含 'label' 和 'score'。
              如果 'transformers' 库未安装或发生错误，则返回 None。
    """
    if pipeline is None:
        print("错误: 'transformers' 库不可用，无法执行图像分类。")
        return None

    try:
        # 尝试加载一个通用的图像分类模型。你也可以指定其他模型，例如 'google/vit-base-patch16-224'
        # 这里使用默认的 'image-classification' 管道，它会尝试下载一个默认模型
        # 首次运行时可能需要下载模型，请耐心等待
        print(f"""正在加载图像分类模型 (可能需要下载，请耐心等待)... 
请确保您的网络连接正常。""")
        sys.stdout.flush() # Force print to display immediately
        # Adding a timeout parameter to the pipeline creation for model download
        classifier = pipeline("image-classification", use_fast=True, timeout=60) # Increase timeout to 60 seconds
        print("模型加载完成。")
        sys.stdout.flush() # Force print to display immediately

        image = Image.open(image_path)
        print(f"正在分类图像: {image_path}...")
        sys.stdout.flush() # Force print to display immediately
        results = classifier(image)
        print("分类完成。")
        sys.stdout.flush() # Force print to display immediately
        return results
    except FileNotFoundError:
        print(f"错误: 无法找到图片 {image_path}。")
        sys.stdout.flush() # Ensure error is displayed
        return None
    except Exception as e:
        print(f"图像分类时发生错误: {e}")
        sys.stdout.flush() # Ensure error is displayed
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # 这是一个简单的测试用例
    # 请确保在运行前有一个名为 'test_image.jpg' 的图片在相同目录下
    # 或者修改为你的图片路径
    test_image_path = 'your.png' # 假设你的图片是your.png
    print(f"尝试对图片 {test_image_path} 进行分类...")
    sys.stdout.flush() # Ensure test message is displayed
    classification_results = classify_image(test_image_path)

    if classification_results:
        print("\n分类结果:")
        for res in classification_results:
            print(f"  标签: {res['label']}, 置信度: {res['score']:.4f}")
    else:
        print("未能获取分类结果。")
    sys.stdout.flush() # Ensure final results are displayed 