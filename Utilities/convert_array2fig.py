from PIL import Image
import os

file_dir = './'


def get_file_list(file_dir):
    for root, dirs, files in os.walk(file_dir):
        pass
    return files


data_list = get_file_list(file_dir)


def convert_array2fig(filename: str):
    with open("2000test/" + filename) as f:
        img = Image.new("RGB", (64, 64), (255, 255, 255))
        data = f.readlines()
        for i in range(len(data)):
            if int((data[i].strip())) == 3:
                row = i // 64
                col = i % 64
                img.putpixel((col,row), (0, 0, 0))

        img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM).save("./" + filename.split(".")[0] + ".png", dpi=(600, 600))
        # img.show()




for name in data_list:
    convert_array2fig(name)
