# 分类网络

## 经验要点

1. 训练时，不要一开始堆叠所有，模型和正则化、由简入繁
2. ToTensor 各像素点值除以了255
3. Normalize  进行了归一化
4. DataLoader 使用num_worker和pin_memory来进行数据异步加载

## 结果

1. 数据增强，							acc：0.691
2. 数据增加，最后一层加dropout，效果不大
3. weigtht_decay 0.001 

