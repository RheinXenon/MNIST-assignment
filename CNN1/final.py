import tkinter as tk
from PIL import Image, ImageDraw
from main import final_mode

class DrawApp:
    def __init__(self, root, width=280, height=280):
        self.root = root
        self.width = width
        self.height = height

        # 创建一个画布
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="black")  # 修改背景为黑色
        self.canvas.pack()

        # 创建一个图片对象，用于绘制
        self.image = Image.new("RGB", (self.width, self.height), "black")  # 修改背景为黑色
        self.draw = ImageDraw.Draw(self.image)

        # 初始化鼠标绘制状态
        self.last_x = None
        self.last_y = None

        # 创建一个标签用于显示预测结果
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack()

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # 绑定键盘按键事件，检测 S 键
        self.root.bind("<KeyPress-s>", self.predict_digit)

    def paint(self, event):
        """鼠标拖动时绘制"""
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 在两个鼠标位置之间绘制线条
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill="white", capstyle=tk.ROUND, smooth=True)  # 修改画笔为白色
            # 在图片上画出相同的线条
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=5)  # 修改画笔为白色
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        """鼠标松开时重置位置"""
        self.last_x = None
        self.last_y = None

    def save_image(self, filename="drawing.png"):
        """保存绘制的图像到文件"""
        self.image.save(filename)

    def clear_canvas(self):
        """清空画布"""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.width, self.height), "black")  # 修改背景为黑色
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self, event):
        """按下 S 键时保存图像并调用数字预测"""
        filename = "drawing.png"
        self.save_image(filename)  # 保存图像

        # 调用 main.py 中的 final 方法进行预测
        predicted_label, confidence = final_mode()

        # 在窗口中显示预测结果
        self.result_label.config(text=f"预测数字: {predicted_label}, 置信度: {confidence:.4f}")

        # 清空画板
        self.clear_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("绘制数字图像")

    app = DrawApp(root)

    # 启动主事件循环
    root.mainloop()
