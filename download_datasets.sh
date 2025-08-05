#!/bin/bash

cd Datasets

# ✅ 下载 MNIST 数据集 (使用 pytorch 自动下载)
echo "🔄 正在下载 MNIST 数据集..."
if [ ! -d "MNIST" ]; then
    mkdir MNIST
    cd MNIST
    echo "📦 使用 Python 自动下载 MNIST 数据集..."
    /usr/bin/python3 -c "
import torchvision.datasets as datasets
datasets.MNIST(root='.', download=True)
"
    echo "✅ MNIST 数据集下载完成！"
    cd ..
else
    echo "✅ MNIST 数据集已存在，跳过下载。"
fi

# ✅ 下载 tiny-imagenet-200 数据集
echo "🔄 正在下载 tiny-imagenet-200 数据集..."
if [ ! -d "tiny-imagenet-200" ]; then
    wget --continue http://cs231n.stanford.edu/tiny-imagenet-200.zip
    echo "🗜️ 解压 tiny-imagenet-200.zip ..."
    unzip -q tiny-imagenet-200.zip
    rm tiny-imagenet-200.zip
    echo "✅ 解压完成，正在重新组织 val 文件夹..."

    # 🔄 重新组织 val 文件夹
    cd tiny-imagenet-200/val
    mkdir images_split
    awk '{print $1}' val_annotations.txt | sort | uniq | while read class; do
        mkdir -p images_split/$class
    done
    while IFS=$'\t' read -r file class _; do
        mv images/$file images_split/$class/
    done < val_annotations.txt
    rm -r images
    mv images_split images
    echo "✅ val 文件夹已成功重新组织！"
    cd ../..

else
    echo "✅ tiny-imagenet-200 数据集已存在，跳过下载。"
fi

# 设置数据集路径
ZIP_FILE="PhC-C2DH-U373_train.zip"
EXTRACT_DIR="PhC-C2DH-U373/Phc-C2DH-U373_train"

# 检查压缩包是否存在
if [ -f "$ZIP_FILE" ]; then
    echo "找到数据集压缩包：$ZIP_FILE"
else
    echo "未找到 $ZIP_FILE,请检查路径是否正确。"
    exit 1
fi

# 创建解压目录
mkdir -p "$EXTRACT_DIR"

# 解压数据集
echo "正在解压数据集到 $EXTRACT_DIR ..."
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"

# 检查解压是否成功
if [ $? -eq 0 ]; then
    echo "✅ 训练集解压完成，数据保存在：$EXTRACT_DIR"
else
    echo "❌ 训练集解压失败，请检查压缩包是否损坏。"
    exit 1
fi

echo "🎉 所有数据集已成功下载并准备好！"
