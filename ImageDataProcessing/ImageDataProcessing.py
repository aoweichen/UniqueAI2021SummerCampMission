# 导入要用的库


class ImageDataProcessing:
    """
        图像数据处理
        传入地址list，从前往后一次对图片进行不同处理
    """
    def __init__(self, path):
        self.path = path

    def rgb2hsl(self):
        # 实现rgb到hsl的转换功能
        # 待填, 对self.path[0]操作
        # 请分别打印self.path[0]的rgb和hsl像素值
        pass

    def vague(self):
        # 实现将图片变模糊功能
        # 待填，对self.path[1]进行操作
        # 请分别打印self.path[1]的模糊前和模糊后的图像
        pass

    def noise_reduction(self):
        # 实现将图片降噪功能
        # 待填，对self.path[2]进行操作
        # 请分别打印self.path[2]的降噪前和降噪后的图像
        pass

    def edge_extraction(self):
        # 实现边缘提取功能
        # 待填，对self.path[3]进行操作
        # 请分别打印self.path[3]的边缘提取前后的图像
        pass

    def brightness_adjustment(self):
        # 实现亮度调整功能
        # 待填，对self.path[4]进行操作
        # 请分别打印self.path[4]的原图，变亮后图像，变暗后图像
        pass

    def rotate(self):
        # 实现旋转功能
        # 待填，对self.path[5]进行操作
        # 请分别打印self.path[5]的原图，旋转任意角度后图像
        pass

    def flip_horizontally(self):
        # 实现水平翻转功能
        # 待填，对self.path[6]进行操作
        # 请分别打印self.path[6]的原图，水平翻转后图像
        pass

    def cutting(self):
        # 实现裁切功能
        # 待填，对self.path[7]进行操作
        # 请分别打印self.path[7]的原图，裁切后图像
        pass

    def resize(self):
        # 实现调整大小功能
        # 待填，对self.path[8]进行操作
        # 请分别打印self.path[8]的原图，调整任意大小后图像
        pass

    def normalization(self):
        # 实现归一化功能
        # 待填，对self.path[9]进行操作
        # 请分别打印self.path[9]的原图，归一化后图像
        pass

    def fit(self):
        self.rgb2hsl()
        self.vague()
        self.noise_reduction()
        self.edge_extraction()
        self.brightness_adjustment()
        self.rotate()
        self.flip_horizontally()
        self.cutting()
        self.resize()
        self.normalization()


if __name__ == '__main__':
    ImageDataProcessing(["pics/0.jpg", "pics/3.jpg", "pics/2.jpg", "pics/8.jpg", "pics/9.jpg", "pics/1.jpg",
                         "pics/7.jpg", "pics/6.jpg", "pics/5.jpg", "pics/4.jpg"]).fit()
