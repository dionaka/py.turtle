import turtle as t
import math
from time import sleep
import cv2
import numpy as np
from PIL import Image # Import Pillow library here, as it's used in mode 5
import os # Import the os module
import sys # Import sys module here for stdout.flush()

# Removed: try...except ImportError for image_classifier
# from image_classifier import classify_image # Import the new module (moved)
sys.stdout.flush() # Ensure any pre-loop output is displayed immediately

def load_image_for_mode(image_path):
    img_color_local = None
    img_pillow_local = None
    img_width_local = 0
    img_height_local = 0
    try:
        # Try loading with OpenCV
        img_color_local = cv2.imread(image_path, 1)
        if img_color_local is None:
             print(f"警告: OpenCV 未能加载图片 {image_path}。")
        else:
             print(f"OpenCV 成功加载图片 {image_path}。")
        sys.stdout.flush()

        # Try loading with Pillow
        img_pillow_local = Image.open(image_path)
        img_pillow_local = img_pillow_local.convert('RGB')
        img_width_local, img_height_local = img_pillow_local.size
        print(f"Pillow 成功加载图片 {image_path}，尺寸: {img_width_local}x{img_height_local}。")
        sys.stdout.flush()
        return img_color_local, img_pillow_local, img_width_local, img_height_local
    except FileNotFoundError:
        print(f"错误: 无法找到图片 {image_path}。请确保文件存在。")
        sys.stdout.flush()
        return None, None, 0, 0
    except Exception as e:
        print(f"加载图片 {image_path} 时发生错误: {e}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return None, None, 0, 0

while True:
    # Add mode selection
    print("请选择绘画模式:")
    print("1. 使用 OpenCV 提取轮廓并绘制")
    print("2. 直接复制像素颜色 (从中心向外)")
    print("3. 直接复制像素颜色 (斐波那契螺旋)")
    print("4. 直接复制像素颜色 (方形螺旋)")
    print("5. 将图片转换为字符画并输出到控制台")
    print("6. 背景移除") # Add background removal option
    print("7. 退出") # Add exit option
    print("8. 图像分类") # Add image classification option
    print("9. 傅里叶平滑一笔画") # Add Fourier smoothing one-stroke drawing option
    print("请选择模式编号 (1, 2, 3, 4, 5, 6, 7, 8 或 9): ")
    sys.stdout.flush() # Add flush here to ensure menu is displayed immediately
    mode = input()
    print(f"DEBUG: 用户输入的模式是: '{mode}'") # 新增调试打印
    sys.stdout.flush() # 确保此调试信息立即显示

    if mode == '7': # Check for exit option
        print("程序退出。")
        break # Exit the loop

    # Initialize image variables to None/0 for current loop iteration
    img_color = None
    img_pillow = None
    img_width = 0
    img_height = 0
    image_path = 'your.png' # Define image path for current loop iteration

    # --- Mode 6: Background Removal ---
    if mode == '6': # Changed to if from elif to be first in this chain
        print("您选择了背景移除功能。")
        
        # Add a message indicating potential model download/processing here
        print("这可能需要一些时间，特别是第一次运行。请耐心等待...")
        sleep(0.1) # Add a small sleep here to ensure message is displayed before potentially long process

        try:
            import rembg
            print("检测到已安装 rembg 库。")
            
            # --- Background Removal Logic Starts Here ---
            while True: # Add an inner loop for background removal attempts
                input_path = input("请输入要移除背景的图片路径 (e.g., input.png): ")
                output_path = os.path.join(os.path.dirname(__file__), 'output.png') # Set output path to script directory + output.png
                print(f"结果将保存到: {output_path}")

                # Check if input file exists
                try:
                    input_image = Image.open(input_path)
                    break # Exit the inner loop if file is found
                except FileNotFoundError:
                    print(f"错误: 未找到文件 {input_path}。请确保文件存在并重新输入。")

            # Only proceed if input_image was successfully loaded (guaranteed by the break above)
            print("请选择输出背景类型:")
            print("1. 透明背景 (默认)")
            print("2. 指定颜色背景")
            bg_choice = input("请输入选项 (1 或 2): ").strip()

            print("正在移除背景...")
            # sleep(0.1) # Removed sleep from here
            output_image = rembg.remove(input_image)

            if bg_choice == '2':
                color_input = input("请输入背景颜色 (R,G,B 格式, e.g., 255,255,255 表示白色): ")
                try:
                    r, g, b = map(int, color_input.split(','))
                    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                         raise ValueError("颜色值超出范围 (0-255)。")
                    background_color = (r, g, b)
                    background = Image.new('RGB', output_image.size, background_color)
                    background.paste(output_image, (0, 0), output_image) # Use output_image as mask
                    final_image = background
                except ValueError as e:
                    print(f"无效的颜色输入: {e}")
                    print("将使用透明背景保存。")
                    final_image = output_image # Revert to transparent if color input is bad
            else:
                final_image = output_image
                print("将使用透明背景保存。")

            try:
                if final_image.mode == 'RGBA' and bg_choice != '2':
                     final_image.save(output_path, format='PNG')
                else:
                     final_image.convert('RGB').save(output_path)
                print(f"背景移除完成，结果已保存到: {output_path}")
            except Exception as save_e:
                print(f"保存文件时发生错误: {save_e}")
                import traceback
                traceback.print_exc()
            # --- Background Removal Logic Ends Here ---

            # After successful background removal, return to the main mode selection loop
            continue

        except ImportError:
            print("未检测到 rembg 库。执行背景移除功能需要安装此库。")
            install_rembg = input("是否现在安装 rembg 库？(y/n): ").lower()
            if install_rembg == 'y':
                print("请等待安装完成后，重新运行脚本并选择模式 6。")
                # The actual installation command will be proposed via a separate tool call after this edit.
            else:
                print("取消安装 rembg 库。无法执行背景移除功能。")

    # --- Mode 8: Image Classification ---
    elif mode == '8':
        print("DEBUG: 进入模式8处理块。")
        sys.stdout.flush() # Ensure this message is displayed immediately
        print("您选择了图像分类功能。")
        sys.stdout.flush() # Ensure this message is displayed immediately

        # Load image for mode 8
        img_color, img_pillow, img_width, img_height = load_image_for_mode(image_path)
        if img_pillow is None: # Image loading failed for mode 8
            continue # Go back to main menu

        # Moved import here to avoid delay at startup
        try:
            from image_classifier import classify_image
            print("DEBUG: 成功导入 image_classifier 模块。")
            sys.stdout.flush() # Ensure immediately display
        except ImportError:
            print("警告: 未能导入 image_classifier 模块。请确保 image_classifier.py 文件存在于相同目录下。")
            sys.stdout.flush() # Ensure warning immediately display
            classify_image = None
            sleep(1) # Add delay to ensure warning is visible

        if classify_image and img_pillow is not None:
            print("正在进行图像分类...")
            sys.stdout.flush() # Ensure immediately display
            # Pass the path to the function, as it internally opens the image
            classification_results = classify_image(image_path)
            if classification_results:
                print("\n图像分类结果:")
                for res in classification_results:
                    print(f"  标签: {res['label']}, 置信度: {res['score']:.4f}")
                sys.stdout.flush() # Ensure immediately display
            else:
                print("未能获取图像分类结果。")
                sys.stdout.flush() # Ensure immediately display
            sleep(1) # Add delay to ensure results or failure message visible
        elif not img_pillow: # If img_pillow is None (image not loaded)
            print("错误: 无法执行图像分类，因为图片未能成功加载。")
            sys.stdout.flush() # Ensure immediately display
            sleep(1) # Add delay to ensure error visible
        else: # If classify_image is None (module not imported)
            print("错误: 图像分类模块不可用。请确保 'image_classifier.py' 文件存在且依赖库已安装。")
            sys.stdout.flush() # Ensure immediately display
            sleep(1) # Add delay to ensure error visible
        continue # Go back to the main menu after classification

    # --- Mode 9: Fourier Smoothing One-Stroke Drawing ---
    elif mode == '9':
        print("您选择了傅里叶平滑一笔画功能。")
        sys.stdout.flush()

        # Load image for mode 9
        img_color, img_pillow, img_width, img_height = load_image_for_mode(image_path)
        if img_pillow is None: # Image loading failed for mode 9
            continue # Go back to main menu

        try:
            import scipy.fft as fft
            print("DEBUG: 成功导入 scipy.fft 模块。")
            sys.stdout.flush()
        except ImportError:
            print("警告: 未能导入 scipy 库的 fft 模块。请确保已安装 scipy 库。")
            sys.stdout.flush()
            sleep(1)
            continue # Go back to main menu if scipy is not available

        # Add a prompt for number of Fourier components
        try:
            num_components_str = input("请输入傅里叶分量数量 (建议 20-200, 更多分量可获得更精细的线条): ").strip()
            num_components = int(num_components_str)
            if not (1 <= num_components <= 10000): # Allow a wide range but warn
                print("警告: 傅里叶分量数量应在 1 到 10000 之间，将使用 100。")
                num_components = 100
        except ValueError:
            print("无效的分量数量输入，默认为 100。")
            num_components = 100

        # --- Turtle Setup for Mode 9 (similar to mode 1-4) ---
        turtle_speed = 0 # Default to fastest speed
        t.speed(turtle_speed)

        print("请选择背景颜色 (black(0) 或 white(1)): ")
        bg_color = input().lower() # Read user input and convert to lowercase

        if bg_color == '1':
            t.Screen().bgcolor("white")
            print("Turtle 窗口背景已设置为白色。")
        else:
            t.Screen().bgcolor("black") # Default to black
            print("无效的背景颜色输入，默认为黑色。")

        t.mode('standard')
        t.color('blue') # Default pen color
        t.pensize(2)
        t.colormode(255) # Set color mode to 255

        # Define global variables and initialize (only if Turtle is needed)
        auto_mul = 1.0 
        global_offset_x = 0
        global_offset_y = 0

        # Set turtle window to full screen (100% of screen width and height)
        t.setup(width=1.0, height=1.0, startx=None, starty=None)

        # Get the actual pixel dimensions of the turtle canvas
        screen_width = t.window_width()
        screen_height = t.window_height()

        # Recalculate scale and offset based on the image dimensions
        def calculate_scale_and_offset_for_mode9(original_width, original_height, screen_width, screen_height):
            global auto_mul, global_offset_x, global_offset_y

            if screen_width > 0 and screen_height > 0:
                scale_w = screen_width / original_width
                scale_h = screen_height / original_height
                auto_mul = min(scale_w, scale_h) * 0.95 # Leave a small margin
                
                global_offset_x = -screen_width / 2.0
                global_offset_y = screen_height / 2.0
            else:
                auto_mul = 0.5
                global_offset_x = -original_width * auto_mul / 2
                global_offset_y = original_height * auto_mul / 2

        calculate_scale_and_offset_for_mode9(img_width, img_height, screen_width, screen_height)
        print(f"使用的缩放倍数 auto_mul: {auto_mul}")
        print(f"使用的偏移量 global_offset_x: {global_offset_x}, global_offset_y: {global_offset_y}")

        # --- Fourier Smoothing and Drawing Logic ---
        print("正在提取轮廓并进行傅里叶平滑...")
        sys.stdout.flush()

        # Convert to grayscale for contour detection
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("未找到任何轮廓。请尝试其他图片或模式。")
            sys.stdout.flush()
            t.done()
            continue

        # Select the largest contour (simplistic approach for 'one-stroke')
        largest_contour = max(contours, key=cv2.contourArea)
        print(f"已选择最大轮廓，包含 {len(largest_contour)} 个点。")

        # Reshape contour points to complex numbers (x + iy)
        complex_contour = np.array([point[0][0] + 1j * point[0][1] for point in largest_contour])

        # Apply Fourier Transform
        fourier_coeffs = fft.fft(complex_contour)

        # Zero out high-frequency components for smoothing
        # We keep the first num_components/2 and the last num_components/2 components
        # and set the rest to zero.
        smoothed_coeffs = np.zeros_like(fourier_coeffs)
        half_components = num_components // 2

        # DC component (index 0)
        smoothed_coeffs[0] = fourier_coeffs[0]

        # Positive frequencies
        smoothed_coeffs[1 : half_components + 1] = fourier_coeffs[1 : half_components + 1]

        # Negative frequencies (from the end of the array)
        if num_components % 2 == 0: # If num_components is even, copy half_components from both ends symmetrically
            smoothed_coeffs[-half_components:] = fourier_coeffs[-half_components:]
        else: # If num_components is odd, copy half_components from positive and half_components+1 from negative
            smoothed_coeffs[-(half_components+1):] = fourier_coeffs[-(half_components+1):]

        # Inverse Fourier Transform to get smoothed contour points
        smoothed_contour = fft.ifft(smoothed_coeffs)
        smoothed_contour_points = np.array([[int(p.real), int(p.imag)] for p in smoothed_contour]).reshape(-1, 1, 2)
        print(f"傅里叶平滑完成，得到 {len(smoothed_contour_points)} 个平滑点。")

        # --- Draw the smoothed contour with Turtle ---
        print("开始绘制平滑轮廓...")
        t.penup()
        
        # Move to the first point of the smoothed contour
        first_point_x, first_point_y = smoothed_contour_points[0][0]
        t.setpos(first_point_x * auto_mul + global_offset_x, -(first_point_y * auto_mul) + global_offset_y)
        t.pendown()

        # Draw the rest of the contour points
        for i, point in enumerate(smoothed_contour_points):
            x, y = point[0]
            t.setpos(x * auto_mul + global_offset_x, -(y * auto_mul) + global_offset_y)
            # Add a small progress update
            if (i + 1) % 100 == 0 or (i + 1) == len(smoothed_contour_points):
                sys.stdout.write(f'\r绘制进度: {((i + 1) / len(smoothed_contour_points)) * 100:.1f}%')
                sys.stdout.flush()
        sys.stdout.write('\n') # New line after completion

        t.penup()
        print("傅里叶平滑一笔画绘制完成。")
        t.done()
        continue # Go back to the main menu

    # If mode is 5, execute the ASCII art conversion logic
    elif mode == '5':
        print("选择模式 5: 将图片转换为字符画并输出到控制台")
        sys.stdout.flush() # Ensure immediately display

        # Load image for mode 5
        img_color, img_pillow, img_width, img_height = load_image_for_mode(image_path)
        if img_pillow is None: # Image loading failed for mode 5
            continue # Go back to main menu
        
        # --- Implement image to ASCII art conversion function ---
        # This function will be called directly in mode 5
        def convert_image_to_ascii(image_pillow):
            # 1. Define ASCII character set (from dark to light)
            # This set can be adjusted as needed for different effects
            # Using a more comprehensive set can give finer detail
            ascii_chars = "@%#*+=-:. " # Simplified character set example
            # More complete character set example (uncomment to use):
            # ascii_chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,."^`"

            # 2. Convert image to grayscale
            img_gray_pillow = image_pillow.convert('L')

            # 3. Resize image to fit console (set fixed width, calculate height proportionally)
            # Assume target width is 100 characters
            new_width = 100
            original_width, original_height = img_gray_pillow.size
            # Consider character aspect ratio, character height is usually about twice the width, so adjust ratio
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio * 0.5) # 0.5 is an empirical value, can be adjusted
            
            # Avoid very small or very large height
            if new_height == 0: new_height = 1
            # Can add max height limit if new_height > max_console_height: new_height = max_console_height
            
            img_resized = img_gray_pillow.resize((new_width, new_height))

            print(f"图片已缩放到 {new_width}x{new_height} 像素进行字符转换。")
            sys.stdout.flush() # Ensure immediately display

            # 4. Iterate through resized image pixels and map to ASCII characters
            ascii_art = []
            width, height = img_resized.size
            
            for y in range(height):
                line = []
                for x in range(width):
                    # Get pixel grayscale value (0-255)
                    pixel_value = img_resized.getpixel((x, y))
                    
                    # Map grayscale value to index in the ASCII character set
                    # Grayscale value 0 (black) corresponds to the first character in the set (most dense/darkest)
                    # Grayscale value 255 (white) corresponds to the last character in the set (least dense/brightest)
                    index = int(pixel_value / 255 * (len(ascii_chars) - 1))
                    
                    # Get the corresponding ASCII character
                    char = ascii_chars[index]
                    line.append(char)
                ascii_art.append("".join(line))
            
            return ascii_art
            
        # --- Execute the conversion and output --- #
        if img_pillow is not None: # This check is now redundant due to previous `if img_pillow is None: continue`
            try:
                ascii_result = convert_image_to_ascii(img_pillow)
                # 5. Output the generated ASCII art to the console
                print("\n生成的字符画:\n")
                sys.stdout.flush() # Ensure immediately display
                for line in ascii_result:
                    print(line)
                print("\n字符画输出完成。")
                sys.stdout.flush() # Ensure immediately display
            except Exception as e:
                print(f"生成字符画时发生错误: {e}")
                sys.stdout.flush() # Ensure immediately display
                import traceback
                traceback.print_exc()
        else:
             print("错误: 无法生成字符画，因为图片未能成功加载。") # This line is now effectively dead code
             sys.stdout.flush() # Ensure immediately display
        continue # Go back to the main menu after classification

    # --- Conditional Turtle Setup and Drawing Logic ---
    # This entire block is only for modes 1, 2, 3, 4
    elif mode in ['1', '2', '3', '4']: # Changed to elif from if
        # Load image for turtle modes
        img_color, img_pillow, img_width, img_height = load_image_for_mode(image_path)
        if img_pillow is None: # Image loading failed for turtle modes
            continue # Go back to main menu

        if mode == '1':
            print("选择模式 1: 使用 OpenCV 提取轮廓并绘制")

            # --- Add precision selection for Mode 1 ---
            # Removed direct precision levels (1, 2, 3)
            # Now, ask user for a resolution percentage
            try:
                resolution_percent_str = input("请输入目标分辨率百分比 (例如 50 表示 50%): ").strip()
                resolution_percent = float(resolution_percent_str)
                if not (1 <= resolution_percent <= 1000): # Allow up to 1000% for extreme cases, but warn
                    print("警告: 分辨率百分比应在 1% 到 1000% 之间，将使用 100%。")
                    resolution_percent = 100.0
            except ValueError:
                print("无效的分辨率输入，默认为 100%。")
                resolution_percent = 100.0

            # --- Add pen color mode selection for Mode 1 ---
            print("请选择画笔颜色模式:")
            print("1. 彩色 (根据图片像素颜色)")
            print("2. 单色 (自定义 RGB)")
            pen_color_choice = input("请输入选项 (1 或 2): ").strip()

            monochromatic_rgb = None # Initialize to None
            if pen_color_choice == '2':
                while True:
                    color_input = input("请输入单色画笔颜色 (R,G,B 格式, e.g., 255,0,0 表示红色): ").strip()
                    try:
                        r, g, b = map(int, color_input.split(','))
                        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                            raise ValueError("颜色值超出范围 (0-255)。")
                        monochromatic_rgb = (r, g, b)
                        print(f"画笔单色已设置为: {monochromatic_rgb}")
                        break
                    except ValueError as e:
                        print(f"无效的颜色输入: {e}。请重新输入。")
            else:
                print("画笔颜色模式将为彩色。")
            
            # --- Add fill option for Mode 1 ---
            print("请选择轮廓填充方式:")
            print("1. 填充轮廓 (默认)")
            print("2. 仅绘制轮廓边缘 (不填充)")
            fill_option_choice = input("请输入选项 (1 或 2): ").strip()
            
            fill_contours = True # Default to fill
            if fill_option_choice == '2':
                fill_contours = False
                print("将仅绘制轮廓边缘。")
            else:
                print("将填充轮廓。")

        # turtle_speed = int(input('输入绘画速度 (0-10, 0最快): ')) # 恢复用户输入速度
        turtle_speed = 0 # 移除直接设置为最快速度
        t.speed(turtle_speed)

        # Define block size for pixel modes (2, 3, 4)
        block_size = 30 # Define block size here

        # Add background color selection
        print("请选择背景颜色 (black(0) 或 white(1)): ")
        bg_color = input().lower() # Read user input and convert to lowercase

        if bg_color == '1':
            t.Screen().bgcolor("white")
            print("Turtle 窗口背景已设置为白色。")
        elif bg_color == 'black':
            t.Screen().bgcolor("0")
            print("Turtle 窗口背景已设置为黑色。")
        else:
            if mode == '1':
                t.Screen().bgcolor("white") # Default to black
                print("无效的背景颜色输入，默认为白色。")
            else:
                t.Screen().bgcolor("black") # Default to black
                print("无效的背景颜色输入，默认为黑色。")

        t.mode('standard')
        t.color('blue')
        # t.setup(1000*mul, 1500*mul, 0, 0) # Removed old setup call
        t.pensize(2)
        t.colormode(255) # Set color mode to 255 to accept integer RGB values 0-255

        # Define global variables and initialize (only if Turtle is needed)
        auto_mul = 1.0 
        global_offset_x = 0
        global_offset_y = 0
        
        # --- Define Turtle Drawing Related Functions ---
        # These functions must be defined *after* importing turtle as t
        def tp(x, y):
            # Use calculated global scale and offset for setpos
            t.penup()
            t.setpos(x*auto_mul + global_offset_x, -(y*auto_mul) + global_offset_y)  
            # Note: tp function no longer includes pendown or penup

        def draw_contours(contours, original_color_img, total_contours, pen_color_choice, monochromatic_rgb, fill_contours):
            import sys
            import time

            start_time = time.time()
            print(f"\n开始绘制 {total_contours} 个轮廓。")

            for i, cnt in enumerate(contours):
                if len(cnt) < 2:
                    continue
                    
                # Get color from the original color image at the first point of the contour
                x0, y0 = cnt[0][0]
                color_bgr = original_color_img[y0, x0]
                color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])) # Convert BGR to RGB tuple
                
                # Set turtle color based on pen_color_choice
                if pen_color_choice == '2' and monochromatic_rgb:
                    t.fillcolor(monochromatic_rgb)
                    t.pencolor(monochromatic_rgb)
                else:
                    t.fillcolor(color_rgb)
                    t.pencolor(color_rgb) # Set stroke color as well

                # Move to the first point (using tp, pen is up at this point)
                x0, y0 = cnt[0][0]
                t.penup()
                t.setpos(x0*auto_mul + global_offset_x, -(y0*auto_mul) + global_offset_y)  
                
                t.pendown() # After moving to the first point, put the pen down
                
                if fill_contours:
                    t.begin_fill() # 开始填充

                # Draw lines sequentially (use setpos for subsequent points, pen is down)
                for pt in cnt[1:]:
                    x, y = pt[0]
                    t.setpos(x*auto_mul + global_offset_x, -(y*auto_mul) + global_offset_y)
                
                if fill_contours:
                    t.end_fill() # 结束填充
                t.penup()

                # --- Progress Bar and Estimated Time --- #
                if (i + 1) % 10 == 0 or (i + 1) == total_contours:
                    elapsed_time = time.time() - start_time
                    # Avoid division by zero
                    if i + 1 > 0:
                        avg_time_per_contour = elapsed_time / (i + 1)
                        remaining_contours = total_contours - (i + 1)
                        estimated_remaining_time = avg_time_per_contour * remaining_contours
                    else:
                        avg_time_per_contour = 0
                        estimated_remaining_time = 0

                    progress_percent = ((i + 1) / total_contours) * 100
                    bar_length = 50
                    filled_length = int(bar_length * (progress_percent / 100))
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)

                    sys.stdout.write(f'\r[{bar}] {progress_percent:.1f}% - 已绘制轮廓: {i+1}/{total_contours} - 预计剩余时间: {estimated_remaining_time:.1f} 秒')
                    sys.stdout.flush()
            sys.stdout.write('\n') # Move to next line after completion
            sys.stdout.flush()
            print(f"所有轮廓绘制完成。总耗时: {time.time() - start_time:.1f} 秒")

        # Function to calculate and set global scale and offset (only if needed for Turtle)
        def calculate_scale_and_offset(original_width, original_height, screen_width, screen_height):
            global auto_mul, global_offset_x, global_offset_y

            if screen_width > 0 and screen_height > 0:
                scale_w = screen_width / original_width
                scale_h = screen_height / original_height
                auto_mul = min(scale_w, scale_h) * 0.95 # Leave a small margin
                
                # Calculate offset to map image top-left (0,0) to Turtle top-left (-screen_width/2, screen_height/2)
                global_offset_x = -screen_width / 2.0
                global_offset_y = screen_height / 2.0  # Y-axis is positive upwards in Turtle

                # print(f"根据屏幕和图片尺寸自动计算的缩放倍数 auto_mul: {auto_mul}") # REMOVED
                # print(f"计算的偏移量 global_offset_x: {global_offset_x}, global_offset_y: {global_offset_y}") # REMOVED
            else:
                # Fallback if screen dimensions are not available
                auto_mul = 0.5 # Default value
                global_offset_x = -original_width * auto_mul / 2 # Simple top-left estimation
                global_offset_y = original_height * auto_mul / 2 # Simple top-left estimation (Y-axis inverted)
                # print("警告：无法获取屏幕尺寸，使用默认缩放倍数 0.5 和简单左上角估算。请确保图片不太大。") # REMOVED

        # New function to draw a single filled pixel block
        def draw_pixel_block(turtle_pos_x_tl, turtle_pos_y_tl, block_color, draw_width, draw_height):
            t.penup()
            t.goto(turtle_pos_x_tl, turtle_pos_y_tl) # Move to the top-left corner of the block

            t.fillcolor(block_color)
            t.pencolor(block_color)

            t.pendown()
            t.begin_fill()
            
            # Draw rectangle from top-left
            t.setheading(0) # Point right
            t.forward(draw_width)
            t.setheading(270) # Point down (since Y is up in Turtle)
            t.forward(draw_height)
            t.setheading(180) # Point left
            t.forward(draw_width)
            t.setheading(90) # Point up
            t.forward(draw_height)

            t.end_fill()
            t.penup()

        # New function to process image and extract block information for Turtle modes (2, 3, 4)
        def process_image_blocks(img_pillow, block_size, auto_mul, img_width, img_height):
            # Store block information in a dictionary for quick lookup by block image coordinates
            block_dict = {}
            # Map image indices back to Turtle coordinates and other drawing info
            img_to_turtle_map = {}

            print("开始提取图片块信息 (用于 Turtle 模式)...")

            for img_y in range(0, img_height, block_size):
                for img_x in range(0, img_width, block_size):
                    # Get the current block area
                    x_end = min(img_x + block_size, img_width)
                    y_end = min(img_y + block_size, img_height)
                    if x_end <= img_x or y_end <= img_y:
                        continue

                    # Calculate the center of the block in image pixel coordinates (for potential sorting/path calculation only)
                    block_center_x_img = img_x + (x_end - img_x) / 2.0
                    block_center_y_img = img_y + (y_end - img_y) / 2.0
                    
                    # Calculate distance from block center to image center (for mode 2 sorting)
                    dist_to_center = math.sqrt((block_center_x_img - img_width/2.0)**2 + (block_center_y_img - img_height/2.0)**2)

                    # Get the average color of the block
                    try:
                        block_img = img_pillow.crop((img_x, img_y, x_end, y_end))
                        block_np = np.array(block_img)
                        if block_np.size == 0:
                           avg_color_rgb = (0, 0, 0) # Default to black if block is empty
                        elif len(block_np.shape) == 2:
                             avg_color_np = [np.mean(block_np)]*3
                             avg_color_rgb = (int(avg_color_np[0]), int(avg_color_np[1]), int(avg_color_np[2]))
                        elif block_np.shape[2] == 4: # Handle potential alpha channel
                             avg_color_np = np.mean(block_np[:, :, :3], axis=(0, 1)) # Average only RGB channels
                             avg_color_rgb = (int(avg_color_np[0]), int(avg_color_np[1]), int(avg_color_np[2]))
                        else: # RGB (3 channels)应该是block_np.shape[2] == 3，这里修正一下
                             avg_color_np = np.mean(block_np, axis=(0, 1))
                             avg_color_rgb = (int(avg_color_np[0]), int(avg_color_np[1]), int(avg_color_np[2]))
                    except Exception as color_e:
                        print(f"获取块颜色时发生错误: {color_e}")
                        avg_color_rgb = (128, 128, 128) # Default to gray on error

                    # Calculate the top-left position of the block in Turtle coordinates (relative to Turtle center 0,0)
                    # Note: This is relative to Turtle's 0,0, NOT the top-left of the screen.
                    # The calculate_scale_and_offset already sets global_offset to map image top-left (0,0) to Turtle top-left.
                    # The drawing loop will use t.goto() with the adjusted position (img_x*auto_mul + global_offset_x, -(img_y*auto_mul) + global_offset_y)
                    # Let's store the necessary info to calculate this inside the drawing loop for clarity,
                    # or store the final Turtle coordinate directly.
                    # Storing image top-left (img_x, img_y) and block size, plus color, is cleaner.

                    # Store the block info with its image grid indices
                    block_ix = int(round(img_x / block_size)) # Use round for better mapping to grid indices
                    block_iy = int(round(img_y / block_size)) # Use round for better mapping to grid indices

                    block_dict[(block_ix, block_iy)] = {
                        'img_x': img_x,
                        'img_y': img_y,
                        'width': x_end - img_x,
                        'height': y_end - img_y,
                        'color': avg_color_rgb,
                        'dist_to_center': dist_to_center # Keep for mode 2 sorting
                    }
                    # For drawing, we need the Turtle position and draw size
                    # Turtle pos = (img_x*auto_mul + global_offset_x, -(img_y*auto_mul) + global_offset_y)
                    # Draw size = (block_width * auto_mul, block_height * auto_mul)
                    # --- Modified calculation to map image center to Turtle (0,0) ---
                    # Turtle pos = ((img_x - img_width/2.0) * auto_mul, -(img_y - img_height/2.0) * auto_mul)
                    img_to_turtle_map[(block_ix, block_iy)] = (\
                        (img_x - img_width/2.0) * auto_mul,\
                        -(img_y - img_height/2.0) * auto_mul,\
                        avg_color_rgb,\
                        (x_end - img_x) * auto_mul,\
                        (y_end - img_y) * auto_mul
                    )
            
            print(f"已收集并处理 {len(block_dict)} 个图片块信息 (用于 Turtle 模式)。")
            max_block_ix = int(math.ceil(img_width / block_size)) -1
            max_block_iy = int(math.ceil(img_height / block_size)) -1
            print(f"最大块索引: ({max_block_ix}, {max_block_iy}) (用于 Turtle 模式)")

            return block_dict, img_to_turtle_map, max_block_ix, max_block_iy

        # Process image and perform actions based on selected mode (for modes 1-4)
        # Ensure both OpenCV and Pillow images are loaded for modes 1-4 processing that might use both
        if img_color is not None and img_pillow is not None:
            # Declare global variables here at the very beginning of the block before they are assigned
            # global auto_mul, global_offset_x, global_offset_y # REMOVED, declared in function

            # print("图片your.png读取成功。") # REMOVED
            original_height, original_width = img_color.shape[:2] # Get dimensions from color image
            # img_width, img_height = img_pillow.size # Get dimensions from Pillow image - Already done outside

            # Convert color image to grayscale for contour detection (only needed for mode 1, but can do it early if needed)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # --- Common Image Loading and Block Info Extraction (for modes 2, 3, 4) ---
            # Only perform these steps if not in mode 5 (which is already checked by the outer if block)

            # Set turtle window to full screen (100% of screen width and height)
            t.setup(width=1.0, height=1.0, startx=None, starty=None)

            # Get the actual pixel dimensions of the turtle canvas
            screen_width = t.window_width()
            screen_height = t.window_height()
            # print(f"Turtle窗口尺寸: {screen_width}x{screen_height}") # REMOVED

            try:
                # img_pillow should be loaded by the common loading block outside the if mode != '5' block
                # Dimensions img_width, img_height are also available from there
                print(f"Pillow 读取图片成功，尺寸: {img_width}x{img_height} (用于 Turtle 模式)") # This print is common for modes 2,3,4

                # Calculate and set global scale and offset using the new function
                calculate_scale_and_offset(img_width, img_height, screen_width, screen_height)
                print(f"使用的缩放倍数 auto_mul: {auto_mul}")
                print(f"使用的偏移量 global_offset_x: {global_offset_x}, global_offset_y: {global_offset_y}")

                # Call the new function to process image blocks
                block_dict, img_to_turtle_map, max_block_ix, max_block_iy = process_image_blocks(img_pillow, block_size, auto_mul, img_width, img_height)

            except Exception as e:
                # Handle potential errors during Pillow loading or block processing for Turtle modes
                print(f"图片加载或块信息提取过程中发生错误 (用于 Turtle 模式): {e}")
                import traceback
                traceback.print_exc()
                # If this fails, we probably can't proceed with Turtle modes, so might exit
                # Consider breaking the while loop or returning to menu instead of exit()
                continue # Go back to menu if image/block processing fails for Turtle modes


        # --- Execute Turtle Drawing Logic based on selected mode (1, 2, 3, 4) ---
        if mode == '1':
            
            scale_factor = resolution_percent / 100.0

            # Resize the image based on scale factor BEFORE contour detection
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            if new_width == 0: new_width = 1 
            if new_height == 0: new_height = 1 

            img_color_resized = cv2.resize(img_color, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img_gray_resized = cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2GRAY)
            
            # Update img_width and img_height to reflect the resized dimensions
            img_width = new_width
            img_height = new_height

            # Recalculate scale and offset based on the *resized* image dimensions
            calculate_scale_and_offset(img_width, img_height, screen_width, screen_height) # Use new_width/new_height
            print(f"重新计算的缩放倍数 auto_mul: {auto_mul}")
            print(f"重新计算的偏移量 global_offset_x: {global_offset_x}, global_offset_y: {global_offset_y}")

            # Use the resized grayscale image for contour detection
            # Use simple threshold if monochromatic color choice for cleaner lines, otherwise adaptive threshold
            if pen_color_choice == '2': # Monochromatic choice
                # Use a simple global threshold for cleaner, simpler contours, similar to 1.0.py
                # You might need to adjust the threshold value (e.g., 127) for best results
                _, binary = cv2.threshold(img_gray_resized, 127, 255, cv2.THRESH_BINARY_INV)
                print("使用简单全局阈值进行二值化处理 (单色模式)。")
            else:
                # Keep adaptive threshold for color mode, or if user didn't choose monochromatic
                binary = cv2.adaptiveThreshold(img_gray_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                print("使用自适应阈值进行二值化处理 (彩色模式或默认)。")

            # 3. Edge detection
            edges = cv2.Canny(binary, 50, 150)

            # 4. Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Add print info for contour count - Retained for general info
            print(f"当前分辨率下检测到 {len(contours)} 个轮廓，开始绘制。") # Updated message

            # 5. Draw all contours with turtle
            draw_contours(contours, img_color_resized, len(contours), pen_color_choice, monochromatic_rgb, fill_contours) # Pass the *resized* color image to the drawing function
            
        elif mode == '2':
            print("选择模式 2: 尝试复制像素颜色块")

            # 遍历图片并绘制块
            t.penup()
            print("开始遍历图片块并绘制...") # Add print info
            # block_count = 0 # Add counter

            # --- Draw test marker ---
            # Move pen to window center and draw test dot (B: Pen starts from center)
            t.goto(0, 0)
            t.dot(20, "red")
            print("已将画笔移动到屏幕中心并绘制红色测试点。")
            # --------------------

            # Calculate image center Turtle coordinates as the drawing origin (A: Image content centered)
            # Note: img_width and img_height are available from the common Pillow loading section

            # We want the center of the image (img_width/2, img_height/2) to map to Turtle's (0,0)
            # So the top-left of the image (0,0) should map to (-img_width/2 * auto_mul, img_height/2 * auto_mul)
            # Image (x,y) maps to Turtle ((x - img_width/2) * auto_mul, -(y - img_height/2) * auto_mul)

            # Store block information for sorting by distance (C: Drawing order from center outwards)
            blocks_to_draw = [] # Re-initialize

            # Populate blocks_to_draw from the pre-calculated block_dict
            for (block_ix, block_iy), block_info in block_dict.items():
                 blocks_to_draw.append({
                      'img_ix': block_ix,
                      'img_iy': block_iy,
                      'pos_x': img_to_turtle_map[(block_ix, block_iy)][0], # Get Turtle pos x from map
                      'pos_y': img_to_turtle_map[(block_ix, block_iy)][1], # Get Turtle pos y from map
                      'width': block_info['width'] * auto_mul, # Use img_info width and apply scale
                      'height': block_info['height'] * auto_mul, # Use img_info height and apply scale
                      'color': block_info['color'],
                      'distance': block_info['dist_to_center'] # Keep for initial sorting idea
                 }) # Use img_width and img_height from common section

            print(f"已收集 {len(blocks_to_draw)} 个块的信息用于排序。")

            # Sort blocks by distance from the center (C: Drawing order)
            blocks_to_draw.sort(key=lambda block: block['distance'])
            print("已按距离中心点排序块信息。")

            # Draw sorted blocks
            print("开始按排序顺序绘制块...")
            block_count = 0 # Re-initialize counter for drawing loop
            for block_entry in blocks_to_draw:
                # Get drawing info from the img_to_turtle_map using sorted block indices
                block_ix = block_entry['img_ix']
                block_iy = block_entry['img_iy']

                if (block_ix, block_iy) in img_to_turtle_map:
                    # Get drawing info from the map
                    turtle_pos_x_tl, turtle_pos_y_tl, block_color, draw_width, draw_height = img_to_turtle_map[(block_ix, block_iy)]

                    # Use the new function to draw the block
                    draw_pixel_block(turtle_pos_x_tl, turtle_pos_y_tl, block_color, draw_width, draw_height)

                    block_count += 1 # Count drawn blocks

            print(f"总共绘制了 {block_count} 个块 (按中心向外顺序)。")
            print("模式 2 块状填充绘制完成。")

        elif mode == '3':
            print("选择模式 3: 直接复制像素颜色 (斐波那契螺旋)")

            # 遍历图片并绘制块
            t.penup()
            print("开始遍历图片块并绘制...") # Add print info
            # block_count = 0 # Re-initialized below

            # --- Draw test marker ---
            # Move pen to window center and draw test dot (B: Pen starts from center)
            t.goto(0, 0)
            t.dot(20, "red")
            print("已将画笔移动到屏幕中心并绘制红色测试点。")
            # --------------------

            # Calculate image center Turtle coordinates as the drawing origin (A: Image content centered)
            # Note: img_width and img_height are available from the common Pillow loading section

            # We want the center of the image (img_width/2, img_height/2) to map to Turtle's (0,0)
            # So the top-left of the image (0,0) should map to (-img_width/2 * auto_mul, img_height/2 * auto_mul)
            # Image (x,y) maps to Turtle ((x - img_width/2) * auto_mul, -(y - img_height/2) * auto_mul)

            # Store block information for sorting by distance (C: Drawing order from center outwards)
            blocks_to_draw = [] # Re-initialize

            # Populate blocks_to_draw from the pre-calculated block_dict
            for (block_ix, block_iy), block_info in block_dict.items():
                 blocks_to_draw.append({
                      'img_ix': block_ix,
                      'img_iy': block_iy,
                      'pos_x': img_to_turtle_map[(block_ix, block_iy)][0], # Get Turtle pos x from map
                      'pos_y': img_to_turtle_map[(block_ix, block_iy)][1], # Get Turtle pos y from map
                      'width': block_info['width'] * auto_mul, # Use img_info width and apply scale
                      'height': block_info['height'] * auto_mul, # Use img_info height and apply scale
                      'color': block_info['color'],
                      'distance': block_info['dist_to_center'] # Keep for initial sorting idea
                 }) # Use img_width and img_height from common section

            print(f"已收集 {len(blocks_to_draw)} 个块的信息。")

            # Re-create block_dict and img_to_turtle_map using the populated blocks_to_draw for lookup by indices
            # This step might be redundant if the initial block_dict/map creation is sufficient.
            # Let's simplify and use the block_dict and img_to_turtle_map created earlier for modes 2, 3, 4.
            # No need to re-create them here.
            # block_dict = {} # Clear and repopulate
            # img_to_turtle_map = {} # Clear and repopulate
            # for block in blocks_to_draw:
            #      # Calculate block's top-left image coordinates (approximate from turtle pos)
            #      img_x_tl = block['pos_x'] / auto_mul + img_width / 2.0
            #      img_y_tl = -block['pos_y'] / auto_mul + img_height / 2.0
            #      block_ix = int(round(img_x_tl / block_size)) # Use round
            #      block_iy = int(round(img_y_tl / block_size)) # Use round
            #      block_dict[(block_ix, block_iy)] = {'width': block['width'] / auto_mul, 'height': block['height'] / auto_mul} # Store original image block size
            #      img_to_turtle_map[(block_ix, block_iy)] = (block['pos_x'], block['pos_y'], block['color'], block['width'], block['height']) # Store all drawing info
            # print(f"已重新创建 {len(block_dict)} 个块的查找字典和映射。")

            # Keep track of drawn blocks by their image indices
            drawn_blocks = set() # Re-initialize

            # Determine the starting block (center of the image) in block indices
            # Use img_width and img_height from the common Pillow loading section
            start_ix = int(round(img_width / 2.0 / block_size))
            start_iy = int(round(img_height / 2.0 / block_size))

            # Use a list to maintain the sequence of blocks to visit (acting as a queue)
            blocks_to_visit_queue = []
            if (start_ix, start_iy) in img_to_turtle_map: # Check against img_to_turtle_map as it contains drawable blocks
                 blocks_to_visit_queue.append((start_ix, start_iy))
                 drawn_blocks.add((start_ix, start_iy))
                 print(f"起始块索引: ({start_ix}, {start_iy})")
            else:
                 print("警告: 图片中心块不存在于可绘制块中，尝试寻找最接近的块作为起点。")
                 # Fallback: find the closest existing block to the center
                 min_dist = float('inf')
                 closest_block_key = None
                 # Iterate through drawable blocks in img_to_turtle_map
                 for key in img_to_turtle_map.keys():
                      # Calculate distance in block indices from the center block index
                      dist = math.sqrt((key[0] - start_ix)**2 + (key[1] - start_iy)**2)
                      if dist < min_dist:
                           min_dist = dist
                           closest_block_key = key
                 if closest_block_key:
                      blocks_to_visit_queue.append(closest_block_key)
                      drawn_blocks.add(closest_block_key)
                      start_ix, start_iy = closest_block_key
                      print(f"找到并使用最接近中心的块作为起点: ({start_ix}, {start_iy})")
                 else:
                      print("错误: 未找到任何块可供绘制。")
                      # Skip drawing logic if no blocks found
                      max_blocks_to_draw_mode3 = 0 # Set limit to 0 to skip the loop below

            # Define possible moves (8 directions: N, NE, E, SE, S, SW, W, NW) in grid indices
            moves = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

            print("开始按斐波那契螺旋近似路径绘制块...")
            block_count = 0
            max_blocks_to_draw_mode3 = len(img_to_turtle_map) # Limit drawing to available drawable blocks

            # Turtle starts at center (0,0) for red dot if mode is not 5, then moves to block position for drawing

            while blocks_to_visit_queue and block_count < max_blocks_to_draw_mode3:
                # Take the next block from the queue (simple FIFO for now, could be refined)
                # Using a list as a queue (pop(0) is O(n), better to use deque for performance with large images)
                # For simplicity with current tools, stick to list for now.
                current_ix, current_iy = blocks_to_visit_queue.pop(0)

                # Get block info and draw
                if (current_ix, current_iy) in img_to_turtle_map:
                    # Get drawing info from the map
                    # The img_to_turtle_map for mode 3 only stores (pos_x, pos_y, color).
                    # We need width and height, which are available in block_dict.
                    turtle_pos_x_tl, turtle_pos_y_tl, block_color, draw_width, draw_height = img_to_turtle_map[(current_ix, current_iy)]
                    # In mode 3, the map only stored 3 items. Need to get width/height from block_dict
                    block_info = block_dict[(current_ix, current_iy)]
                    draw_width = block_info['width'] * auto_mul # Scale the original width
                    draw_height = block_info['height'] * auto_mul # Scale the original height

                    # Use the new function to draw the block
                    draw_pixel_block(turtle_pos_x_tl, turtle_pos_y_tl, block_color, draw_width, draw_height)

                    block_count += 1

                    # Find potential next blocks (neighbors)
                    next_blocks_candidates = []
                    for move_dx, move_dy in moves:
                        neighbor_ix = current_ix + move_dx
                        neighbor_iy = current_iy + move_dy

                        # Check if neighbor is within bounds and exists in our drawable block dictionary (img_to_turtle_map)
                        # Need to determine image bounds in terms of block indices
                        # Max indices would be ceil(img_width / block_size) - 1 and ceil(img_height / block_size) - 1
                        max_ix = int(math.ceil(img_width / block_size)) - 1
                        max_iy = int(math.ceil(img_height / block_size)) - 1

                        if (0 <= neighbor_ix <= max_ix and 0 <= neighbor_iy <= max_iy and
                            (neighbor_ix, neighbor_iy) in img_to_turtle_map and
                            (neighbor_ix, neighbor_iy) not in drawn_blocks):
                           
                           next_blocks_candidates.append((neighbor_ix, neighbor_iy))

                    # --- Selection Logic for the next block ---
                    # Choose the neighbor that best follows the spiral path.
                    # This is a simplification; a true spiral follows a specific angle pattern.
                    # For a simple approximation, we can try to favor moving 'outward'
                    # or based on a theoretical spiral angle at the current block's position.

                    # Calculate the position of the current block's center in image coordinates
                    current_img_x_center = (current_ix + 0.5) * block_size
                    current_img_y_center = (current_iy + 0.5) * block_size

                    # Calculate the ideal spiral angle (theta) at the current image center position
                    # Based on r = a * sqrt(theta), theta = (r/a)^2
                    # r is the distance from the image center in image pixels
                    dist_from_img_center = math.sqrt((current_img_x_center - img_width/2.0)**2 + (current_img_y_center - img_height/2.0)**2)
                    
                    # Avoid division by zero if exactly at the center
                    ideal_spiral_angle = 0 # Default angle if at center or calculation fails
                    if dist_from_img_center > 1e-6:
                        # Approximating theta - this is tricky as theta defines r, and r defines theta
                        # A simple way might be to track the 'unwrapped' angle as we move.
                        # Let's simplify and use the angle from the image center to the current block center.
                        angle_to_current_block = math.atan2(-(current_img_y_center - img_height/2.0), current_img_x_center - img_width/2.0) # Use -dy for standard math angle (Y up)

                        # The spiral direction should be roughly perpendicular to the vector from the center,
                        # plus a small outward component.
                        # For a counter-clockwise spiral, this is roughly angle_to_current_block + PI/2
                        # Need to normalize angle to [0, 2*PI) or similar
                        ideal_spiral_angle = angle_to_current_block + math.pi / 2.0
                        # Normalize angle
                        ideal_spiral_angle = ideal_spiral_angle % (2 * math.pi)
                        if ideal_spiral_angle < 0:
                            ideal_spiral_angle += 2 * math.pi


                    def angle_difference(angle1, angle2):
                         diff = abs(angle1 - angle2)
                         return min(diff, 2 * math.pi - diff) # Consider wrapping around 2*PI

                    # Sort candidates based on how well their direction vector from the current block aligns with the ideal spiral angle
                    def calculate_alignment_score(current_ix, current_iy, candidate_ix, candidate_iy, ideal_angle):
                         # Vector from current block center to candidate block center (in block indices)
                         vec_dx = (candidate_ix - current_ix) # * block_size # In block indices
                         vec_dy = (candidate_iy - current_iy) # * block_size # In block indices (Y is down in image coords)
                         
                         # Calculate the angle of this vector. Use -vec_dy because math.atan2 assumes Y is up.
                         candidate_angle = math.atan2(-vec_dy, vec_dx)
                         # Normalize angle
                         candidate_angle = candidate_angle % (2 * math.pi)
                         if candidate_angle < 0:
                             candidate_angle += 2 * math.pi

                         # Score is inverse of angle difference (smaller difference is better)
                         return -angle_difference(ideal_angle, candidate_angle) # Negative for descending sort (smaller diff is better)

                    # Sort candidates. A more sophisticated sort might consider distance from center as a secondary factor.
                    next_blocks_candidates.sort(key=lambda candidate: calculate_alignment_score(current_ix, current_iy, candidate[0], candidate[1], ideal_spiral_angle), reverse=True) # Sort descending by score (ascending by difference)

                    # Add sorted, unvisited neighbors to the front of the queue to prioritize exploring along the spiral
                    # This makes it more of a priority queue/greedy approach than strict FIFO
                    for next_block_key in next_blocks_candidates:
                         if next_block_key not in drawn_blocks:
                            blocks_to_visit_queue.insert(0, next_block_key) # Add to the front
                            drawn_blocks.add(next_block_key)


            print(f"总共绘制了 {block_count} 个块 (按近似斐波那契螺旋路径)。")
            print("模式 3 近似斐波那契螺旋填充绘制完成。")

        elif mode == '4':
            print("选择模式 4: 直接复制像素颜色 (方形螺旋)")

            # 遍历图片并绘制块
            t.penup()
            print("开始遍历图片块并绘制...") # Add print info
            # block_count = 0 # Re-initialized below

            # --- Draw test marker ---
            # Move pen to window center and draw test dot (B: Pen starts from center)
            t.goto(0, 0)
            t.dot(20, "red")
            print("已将画笔移动到屏幕中心并绘制红色测试点。")
            # --------------------

            # Calculate image dimensions and block size (available from common section)
            # img_width, img_height, block_size are available
            # block_dict and img_to_turtle_map are populated

            # Keep track of drawn blocks by their image indices
            drawn_blocks = set()

            # --- Square Spiral Traversal Logic --- #

            # Determine the starting block (center of the image) in block indices
            # Use img_width and img_height from the common section
            start_ix = int(round(img_width / 2.0 / block_size))
            start_iy = int(round(img_height / 2.0 / block_size))
            
            # Initialize position and movement parameters
            current_ix, current_iy = start_ix, start_iy

            # Debug: Print info about the starting block
            print(f"计算出的起始块索引 (近似图片中心): ({start_ix}, {start_iy})")
            if (start_ix, start_iy) in img_to_turtle_map:
                turtle_pos_x_tl, turtle_pos_y_tl, _, draw_width, draw_height = img_to_turtle_map[(start_ix, start_iy)]
                turtle_pos_x_center = turtle_pos_x_tl + draw_width / 2.0
                turtle_pos_y_center = turtle_pos_y_tl - draw_height / 2.0
                print(f"起始块左上角 Turtle 坐标: ({turtle_pos_x_tl:.2f}, {turtle_pos_y_tl:.2f})")
                print(f"起始块中心 Turtle 坐标 (预期绘制起点): ({turtle_pos_x_center:.2f}, {turtle_pos_y_center:.2f})")
            else:
                print(f"警告: 计算出的起始块索引 ({start_ix}, {start_iy}) 不存在于 img_to_turtle_map 中。")

            # Spiral parameters
            step_length = 1
            direction = 0 # 0: Right, 1: Down, 2: Left, 3: Up
            steps_taken_in_dir = 0
            segment_count = 0

            block_count = 0
            # Limit drawing to available drawable blocks
            max_blocks_to_draw = len(img_to_turtle_map)

            # Turtle starts at center (0,0) for red dot, then moves to block center for drawing

            # The spiral should start from the center and move outwards.
            # The drawing loop should handle drawing the current block, then calculating the *next* block.
            # The order of operations was likely the issue in the previous attempt.

            # Use a while loop based on the number of blocks to draw
            while block_count < max_blocks_to_draw:
                # --- Draw the current block if it exists and hasn't been drawn --- #
                # Check if the current block index is within the drawable map and hasn't been drawn.
                # Note: current_ix, current_iy might go outside the original image bounds in later spiral turns,
                # so we must check against the populated img_to_turtle_map.
                if (current_ix, current_iy) in img_to_turtle_map and (current_ix, current_iy) not in drawn_blocks:
                    # Get drawing info from the map
                    turtle_pos_x_tl, turtle_pos_y_tl, block_color, draw_width, draw_height = img_to_turtle_map[(current_ix, current_iy)]

                    # Use the new function to draw the block
                    draw_pixel_block(turtle_pos_x_tl, turtle_pos_y_tl, block_color, draw_width, draw_height)

                    # Mark block as drawn and increment count
                    drawn_blocks.add((current_ix, current_iy))
                    block_count += 1

                    # Debug: Print info after drawing a block
                    # print(f"已绘制块，索引: ({current_ix}, {current_iy}), Turtle 绘制起点: ({turtle_pos_x_tl:.2f}, {turtle_pos_y_tl:.2f})")
                    
                    # Stop if we have drawn all available blocks
                    if block_count == max_blocks_to_draw:
                        # print("已绘制所有可用块，停止遍历。") # Debug print
                        break # Exit the while loop after drawing the last block

                # --- Calculate the next block position and update spiral state --- #
                # This happens *after* attempting to draw the current block.

                # Calculate the next potential block position based on the current direction
                next_ix, next_iy = current_ix, current_iy
                if direction == 0: # Right
                    next_ix += 1
                elif direction == 1: # Down
                    next_iy += 1
                elif direction == 2: # Left
                    next_ix -= 1
                elif direction == 3: # Up
                    next_iy -= 1

                steps_taken_in_dir += 1

                # Check if it's time to change direction
                # This happens when we have taken 'step_length' steps in the current direction.
                if steps_taken_in_dir == step_length:
                    direction = (direction + 1) % 4 # Change direction (0->1, 1->2, 2->3, 3->0)

                    # Increase step length after every two segments are completed.
                    # A segment is completed when we change direction.
                    # The step length sequence is 1, 1, 2, 2, 3, 3, ...
                    segment_count += 1 # Increment segment count after changing direction
                    if segment_count % 2 == 0: # Increase step length after segments 2, 4, 6, ...
                        step_length += 1

                    steps_taken_in_dir = 0 # Reset steps count for the new direction

                # Update the current block index for the *next* iteration.
                # This is where we move to the calculated next position.
                current_ix, current_iy = next_ix, next_iy

                # The loop continues. In the next iteration, it will attempt to draw the block
                # at the new (current_ix, current_iy) if it exists and is not drawn.


            print(f"总共绘制了 {block_count} 个块 (按方形螺旋路径)。")
            print("模式 4 方形螺旋填充绘制完成。")

        # Only call t.done() if a Turtle mode (1-4) was executed
        # This will keep the Turtle window open until manually closed or a new mode is selected
        # If mode 5 is selected, t.done() is skipped, and the while loop continues to the next iteration
        t.done()

    # Handle invalid mode input
    elif mode not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        print("错误：选择的模式无效。")

# The while loop finishes here when mode is '7'.
# Any code here would execute after the loop.
