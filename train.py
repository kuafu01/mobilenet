import os
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v3 import mobilenet_v3_large


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 16#每一批次训练的图片数
    epochs = 5#训练轮数

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),#数据增强，随机裁剪大小224
                                     transforms.RandomHorizontalFlip(),#数据增强，水平方向随机翻转
                                     transforms.ToTensor(),#转化成张量
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),#归一化处理
        "val": transforms.Compose([transforms.Resize(256),#伸缩大小为256
                                   transforms.CenterCrop(224),#中心裁剪为224
                                   transforms.ToTensor(),#转化为张量
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}#归一化

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)#数据集大小

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())#将键值和索引反转，利于对最后的结果索引
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:#样式文件
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))#cpu载入线程数，详见detail explanation1.

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,#随机的从样本中获取个数为batch_size的样本图片
                                               num_workers=nw)#num_workers为加载图像的线程数

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),#路径
                                            transform=data_transform["val"])#测试集预处理函数
    val_num = len(validate_dataset)#测试集大小
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,#不随机抽样
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))#打印训练和测试使用的图片数目

    # create model
    net = mobilenet_v3_large(num_classes=5)#使用的模型和标签数（类别数目）

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v3.pth"#预训练权重
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)#判断是否存在
    pre_weights = torch.load(model_weight_path, map_location='cpu')#加载权重

    # delete classifier weights
   #排除全连接层的权重，以适应新的任务或数据集 详见DE2.
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)


    # freeze features weights-----冻结特征提取部分的权重更新
    for param in net.features.parameters():
        param.requires_grad = False


    net.to(device)  # 设备

    # define loss function----损失函数
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer----优化器，更新可更新的权重
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0#用来保存训练准确率最高的那个模型
    save_path = './MobileNetV3.pth'
    train_steps = len(train_loader)
    # 遍历每一个epoch，进行模型的训练和验证
    for epoch in range(epochs):
        # 将网络设置为训练模式
        net.train()
        running_loss = 0.0  # 用于记录每个epoch的累积损失
        train_bar = tqdm(train_loader, file=sys.stdout)  # 创建一个进度条以显示训练过程

        # 遍历训练数据加载器中的每一个batch
        for step, data in enumerate(train_bar):
            images, labels = data  # 获取当前batch的图像和标签
            optimizer.zero_grad()  # 清空之前的梯度
            logits = net(images.to(device))  # 前向传播：将图像输入网络并获取输出logits
            loss = loss_function(logits, labels.to(device))  # 计算损失函数值
            loss.backward()  # 反向传播：计算梯度
            optimizer.step()  # 更新模型参数

            # 打印统计信息
            running_loss += loss.item()  # 累积损失值
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)  # 更新进度条描述，显示当前epoch和损失值

        # 验证阶段
        net.eval()  # 将网络设置为评估模式（关闭dropout等）
        acc = 0.0  # 用于记录每个epoch的累积正确预测数量
        with torch.no_grad():  # 关闭梯度计算，减少内存消耗并加速计算
            val_bar = tqdm(validate_loader, file=sys.stdout)  # 创建一个进度条以显示验证过程

            # 遍历验证数据加载器中的每一个batch
            for val_data in val_bar:
                val_images, val_labels = val_data  # 获取当前batch的验证图像和标签
                outputs = net(val_images.to(device))  # 前向传播：将验证图像输入网络并获取输出
                predict_y = torch.max(outputs, dim=1)[1]  # 获取预测的类别（取最大值的索引）
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 累积正确预测的数量

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)  # 更新进度条描述，显示当前epoch

        val_accurate = acc / val_num  # 计算验证集上的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))  # 打印当前epoch的训练损失和验证准确率

        # 如果当前epoch的验证准确率高于之前的最佳准确率，则保存模型参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')  # 训练完成


if __name__ == '__main__':
    main()
