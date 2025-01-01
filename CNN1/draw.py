import tkinter as tk
from PIL import Image, ImageDraw
from main import final

class DrawApp:
    def __init__(self, root, width=280, height=280):
        self.root = root
        self.width = width
        self.height = height
        
        # 创建一个画布
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        # 创建一个图片对象，用于绘制
        self.image = Image.new("RGB", (self.width, self.height), "white")
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
            # 在两个鼠标位置之间绘制线条
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill="black", capstyle=tk.ROUND, smooth=True)
            # 在图片上画出相同的线条
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=5)
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        """鼠标松开时重置位置"""
        self.last_x = None
        self.last_y = None

    def get_image(self):
        """返回绘制的图像"""
        return self.image

def save_image(image, filename="drawing.png"):
    """保存图像"""
    image.save(filename)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("绘制数字图像")
    root.bind('<s>', lambda event: save_image(app.get_image(), "drawing.png"))
    app = DrawApp(root)

    # 启动主事件循环
    root.mainloop()
