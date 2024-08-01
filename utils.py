import logging
import os
import matplotlib.pyplot as plt

def logprint(msg):
    print(msg)
    logging.info(msg)

def create_unique_folder(base_folder):
    """
    Creates a folder at the specified path. If the folder already exists, creates a new folder with a suffix '_n'
    where 'n' is the number of times the folder name has been duplicated.
    
    :param base_folder: Path to the base folder
    """
    folder_path = base_folder
    counter = 1
    
    # Check if the folder exists and create a new folder name if it does
    while os.path.exists(folder_path):
        folder_path = f"{base_folder}_{counter}"
        counter += 1
    
    # Create the folder
    os.makedirs(folder_path)
    print(f"Folder created at: {folder_path}")
    return folder_path


def plot_and_save(data_list, title, save_dir, file_name):
    """
    绘制曲线并保存到指定文件夹

    参数:
    data_list (list): 要绘制的数据列表
    title (str): 图像标题
    save_dir (str): 保存文件夹路径
    file_name (str): 保存的文件名
    """
    # 创建保存文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(data_list)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    # 保存图像
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()