import tkinter as tk
from PIL import Image, ImageDraw
import os

class DrawApp:
    def __init__(self, root, width=280, height=280):
        self.root = root
        self.width = width
        self.height = height
        
        # 创建一个画布
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        # 创建一个图片对象，用于绘制 (黑底白字)
        self.image = Image.new("RGB", (self.width, self.height), "black")
        self.draw = ImageDraw.Draw(self.image)

        # 初始化鼠标绘制状态
        self.last_x = None
        self.last_y = None

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        """鼠标拖动时绘制"""
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 在两个鼠标位置之间绘制线条 (白色)
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill="white", capstyle=tk.ROUND, smooth=True)
            # 在图片上画出相同的线条 (白色)
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=10)
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        """鼠标松开时重置位置"""
        self.last_x = None
        self.last_y = None

    def get_image(self):
        """返回绘制的图像"""
        return self.image

    def clear_canvas(self):
        """清空画布和图像"""
        self.canvas.delete("all")  # 清空画布
        self.image = Image.new("RGB", (self.width, self.height), "black")  # 重置图像为黑色背景
        self.draw = ImageDraw.Draw(self.image)

def save_image(image, app, base_filename="drawing"):
    """保存图像并清空画布"""
    i = 1
    while True:
        filename = f"{base_filename}{i}.png"
        if not os.path.exists(filename):
            image.save(filename)
            print(f"图像已保存为 {filename}")
            break
        i += 1
    app.clear_canvas()  # 保存后清空画布

if __name__ == "__main__":
    root = tk.Tk()
    root.title("绘制数字图像")
    app = DrawApp(root)

    # 按下 'S' 保存图片并清空画布
    root.bind('<s>', lambda event: save_image(app.get_image(), app))

    # 启动主事件循环
    root.mainloop()
