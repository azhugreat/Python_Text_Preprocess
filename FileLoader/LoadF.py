# ！coding:utf-8

import os


# 遍历文件夹
class loadFolders(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath):
                yield file_abspath

"""
遍历文件夹中文件
@return catg    文本类别
@return content 文本内容
"""
class loadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:
            catg = folder.split(os.sep)
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    this_file = open(file_path, 'rb')
                    content = this_file.read().decode('utf-8')
                    yield catg, content
                    this_file.close()
