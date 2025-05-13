from PIL import Image

# 替换为你的图片路径
long_imgs = ["output_1.png", "output_2.png", "output_3.png"]
square_img_path = "gt.png"

# 打开所有长条图片
long_images = [Image.open(img) for img in long_imgs]

# 获取统一宽度（假设一致），计算总高度
width = long_images[0].width
total_height = sum(img.height for img in long_images)

# 竖向拼接三张长条图
merged_long = Image.new("RGB", (width, total_height))
y_offset = 0
for img in long_images:
    merged_long.paste(img, (0, y_offset))
    y_offset += img.height

# 打开正方形图，不改变其纵横比，按高度等比例缩放
square_img = Image.open(square_img_path)
sq_ratio = total_height / square_img.height
new_sq_width = int(square_img.width * sq_ratio)
square_img_resized = square_img.resize((new_sq_width, total_height), Image.LANCZOS)

# 横向拼接，左边是缩放后的正方形图，右边是竖向拼接图
final_width = new_sq_width + merged_long.width
final_image = Image.new("RGB", (final_width, total_height))
final_image.paste(square_img_resized, (0, 0))
final_image.paste(merged_long, (new_sq_width, 0))

# 保存结果
final_image.save("final_result.jpg")
print("拼接完成，已保存为 final_result.jpg")
