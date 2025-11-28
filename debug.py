import os
print(os.getcwd())
print(os.path.exists("./dataset"))
print(os.path.exists("./dataset/train"))
print(os.path.exists("./dataset/val"))
print(os.listdir("./dataset/val"))
