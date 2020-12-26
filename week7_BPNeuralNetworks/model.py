import argparse
import pickle
from dataLoder import *
class Model:
    '''
    分为三层：1、输入层：28*28=784图片
            2、隐藏层：1024层
            3、输出层：输出0-9共10种分类
    y=wx+b 更新权重w和偏置值b
    '''
    def __init__(self, args, parameters = None):
        '''
        :param args:输入参数，主要是神经网络各层的维度
        :param parameters:参数，指w,b。如果没有输入则随机生成并进一步训练调整，如果有输入则采用输入
        '''
        self.dimensions=[int(args.inputSize)+1, int(args.hiddenSize)+1,int(args.outputSize)]
        self.distrbute=[[-1/self.dimensions[1], -1/self.dimensions[1]],#隐藏层参数分布
                        [-1/self.dimensions[2], 1/self.dimensions[2]]]#输出层参数分布
        if parameters:
            self.parameters=parameters
        else:
            self.parameters=self.initParameters()

    def pred_laber(self,x):
        #x=self.tanh(x)
        x=np.dot(x,self.parameters[0])
        x=self.tanh(x)
        x=np.dot(x,self.parameters[1])
        x=self.softmax(x)
        return x

    def oneHot(self,label):
        tmp=np.zeros(self.dimensions[2])
        tmp[label]=1
        return tmp

    def lossFunction(self, data, label):
        y_pred=self.pred_laber(data)
        y=self.oneHot(label)

        return np.dot(y-y_pred,y-y_pred)/2

    def gradient(self,data,label):
        data=list(data)
        data.append(1)
        data=np.array(data)
        in1=data#785
        #out1=self.tanh(in1)#785
        out1 = in1
        in2 = np.dot(out1,self.parameters[0])#1025
        out2 = self.tanh(in2)#1025
        in3 = np.dot(out2,self.parameters[1])#10
        out3 = self.softmax(in3)#10
        print(self.lossFunction(data, label))
        y = self.oneHot(label)
        y_pred = out3
        d_param2 = np.outer(out2, np.dot(self.d_softmax(in3), (y_pred-y)))#1025*10
        d_param1 = np.dot(self.d_softmax(in3), (y_pred-y))#10
        d_param1 = np.dot(self.parameters[1], d_param1)#1025
        d_param1 = d_param1*self.d_tanh(in2)#1025
        d_param1 = np.outer(out1, d_param1)
        res = []
        res.append(d_param1)
        res.append(d_param2)
        return res

    def updateParameters(self, data, label, learningRate):
        grad=self.gradient(data, label)
        self.parameters[0] -= learningRate*grad[0]
        self.parameters[1] -= learningRate*grad[1]


    def initParameters(self):
        parame=[]
        for index,value in enumerate(self.distrbute):
            parame.append(self.initPara(index,value))
        return parame

    def initPara(self,index,value):
            return np.random.rand(self.dimensions[index],self.dimensions[index+1])*(value[1]-value[0])+value[0]#二维
    def sigmod(self,x):
        return 1/(1+np.exp(-x))
    def tanh(self,x):
        return np.tanh(x)
    def softmax(self,x):
        z=np.exp(x-x.max())
        return z/z.sum()
    def d_sigmod(self,x):
        return (1-self.sigmod(x))*self.sigmod(x)
    def d_tanh(self,x):
        return 1-np.power(self.tanh(x),2)
    def d_softmax(self,x):
        z=self.softmax(x)
        return np.diag(z)-np.dot(z,z)
if __name__=='__main__':
    parse=argparse.ArgumentParser('输入参数')
    parse.add_argument('--inputSize',help=' dimension of input',default=784)
    parse.add_argument('--hiddenSize',help=' dimension of hidden',default=1024)
    parse.add_argument('--outputSize',help='dimension of output',default=10)
    parse.add_argument('--parameters', help='parameters file', default="parameters.pkl")
    args=parse.parse_args()
    parameters = None
    # 读取参数
    if args.parameters:
        with open('parameters.pkl', 'rb') as file_to_read:
            # 通过pickle的load函数读取data1.pkl中的对象，并赋值给data2
            parameters = pickle.load(file_to_read)
    model = Model(args,parameters)
    data=dataLoder()
    data.readData()
    count = 0
    # for j in range(5):
    # for i in range(len(data.train_label)):
    # #for i in range(10000):
    #     # print(i)
    #     # for j in range(10):
    #     model.updateParameters(data.train_img[i], data.train_label[i], 0.000003)
    print('train_set_size:',len(data.train_label))
    print('test_set_size:',len(data.test_label))
    for i in range(len(data.test_label)):
        y=list(data.test_img[i])
        y.append(1)
        y=np.array(y)
        y=model.pred_laber(y)
        #print(y)
        #print(y.argmax(),data.test_label[i])

        if y.argmax()==data.test_label[i]:
            count+=1
    print('test_right:',count)
    print('test accuracy:',count/len(data.test_label))
    count = 0
    for i in range(len(data.train_label)):
        y=list(data.train_img[i])
        y.append(1)
        y=np.array(y)
        y=model.pred_laber(y)
        #print(y)
        #print(y.argmax(),data.test_label[i])

        if y.argmax()==data.train_label[i]:
            count+=1
    print('train_right:',count)
    print('train accuracy:',count/len(data.train_label))
    with open('parameters.pkl','wb') as code:
        pickle.dump(model.parameters,code,-1)

