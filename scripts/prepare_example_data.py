import os
import requests
import zipfile

def main(url, dataset):
    # 获取脚本所在目录的父目录（项目根目录）
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(root_dir, 'datasets', dataset)
    zip_path = os.path.join(root_dir, 'datasets', f'{dataset}.zip')

    # 创建目标文件夹
    os.makedirs(dataset_dir, exist_ok=True)

    # 下载文件
    print(f'正在下载 {url} ...')
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    # 解压文件
    print(f'正在解压 {zip_path} ...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print(f'解压完成: {dataset_dir}')

if __name__ == '__main__':
    """该脚本将下载并准备示例数据：
        1. BSDS100 用于训练
        2. Set5 用于测试
    """
    # 数据集的下载链接
    urls = [
        'https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/BSDS100.zip',
        'https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/Set5.zip'
    ]
    datasets = ['BSDS100', 'Set5']

    # 下载并解压每个数据集
    for url, dataset in zip(urls, datasets):
        main(url, dataset)
