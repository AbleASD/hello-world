import os
import requests
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plot
class dataLoder:
    '''
    train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
    train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
    t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
    t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
    '''
    def __init__(self):
        self.dataPath=os.path.join(os.getcwd(),'data')
        if not os.path.exists(self.dataPath):
            os.mkdir(self.dataPath)
        self.training_set_images=os.path.join(self.dataPath, 'train-images-idx3-ubyte.gz')
        self.training_set_labels=os.path.join(self.dataPath,'train-labels-idx1-ubyte.gz')
        self.test_set_images=os.path.join(self.dataPath,'t10k-images-idx3-ubyte.gz')
        self.test_set_labels=os.path.join(self.dataPath,'t10k-labels-idx1-ubyte.gz')
        self.downloadData()
        self.readData()

    def downloadfile(self,url,fname):

        if not os.path.exists(fname):
            print(f'downloading {fname}')
            r = requests.get(url)
            with open(fname,'wb') as code:
                code.write(r.content)
        else:
            print(f'{fname}:文件已存在1')


    def downloadData(self):
        trainImageUrl='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        trainLabelUrl='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        testImageUrl='http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        testLabelUrl='http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        #下载
        self.downloadfile(trainImageUrl,self.training_set_images)
        self.downloadfile(trainLabelUrl,self.training_set_labels)
        self.downloadfile(testImageUrl,self.test_set_images)
        self.downloadfile(testLabelUrl,self.test_set_labels)
        #解压
        self.untar(self.training_set_images)
        self.untar(self.training_set_labels)
        self.untar(self.test_set_images)
        self.untar(self.test_set_labels)


    def untar(self,fname):
        try:
            dir=fname.replace('.gz','')
            if not os.path.exists(dir):
                print(f'ungzing {fname}')
                f=gzip.GzipFile(fname)
                with open(dir,'wb+') as code:
                    code.write(f.read())
                f.close()
                print(f'{fname} ungz down')
            print(f'{fname}:文件已存在')

        except Exception as e:
            print(e)
            return False
    def readData(self):
        #训练集图片
        with open(self.training_set_images.replace('.gz',''),'rb') as code:
            magic_number,number_of_image,rows,cols = struct.unpack('>4i',code.read(struct.calcsize('>4i')))
            print(f'train set image:\n  magic number:{magic_number},number of image:{number_of_image},rows：{rows},cols:{cols}')
            self.train_img=np.array(struct.unpack(f'>{number_of_image*rows*cols}B',code.read()),dtype=np.int).reshape(-1,rows*cols)/255
        #训练集标签
        with open(self.training_set_labels.replace('.gz',''),'rb') as code:
            magic_number,number_of_items=struct.unpack('>2i',code.read(struct.calcsize('>2i')))
            print(f'train set label:\n  magic number:{magic_number},number of items:{number_of_items}')
            self.train_label=np.array(struct.unpack(f'>{number_of_items}B',code.read(struct.calcsize(f'>{number_of_items}B'))),
                                      np.int)
        #测试集图片
        with open(self.test_set_images.replace('.gz',''),'rb') as code:
            magic_number,number_of_image,rows,cols = struct.unpack('>4i',code.read(struct.calcsize('>4i')))
            print(f'test set image:\n  magic number:{magic_number},number of image:{number_of_image},rows：{rows},cols:{cols}')
            self.test_img=np.array(struct.unpack(f'>{number_of_image*rows*cols}B',code.read()),dtype=np.int).reshape(-1,rows*cols)/255
        #测试集标签
        with open(self.test_set_labels.replace('.gz',''),'rb') as code:
            magic_number,number_of_items=struct.unpack('>2i',code.read(struct.calcsize('>2i')))
            print(f'train set label:\n  magic number:{magic_number},number of items:{number_of_items}')
            self.test_label=np.array(struct.unpack(f'>{number_of_items}B',code.read(struct.calcsize(f'>{number_of_items}B'))),
                                      np.int)
if __name__=='__main__':
    data=dataLoder()
    #test=data.train_img[0].reshape(28,28) * 255
    test=data.test_img[0].reshape(28, 28) * 255

    #print(data.train_label[0])
    print(data.test_label[0])
    # print(data.train_img[0])
    # print(data.test_img[0])
    plot.title('test')
    plot.imshow(test,cmap='gray')
    plot.show()
    print('done')


