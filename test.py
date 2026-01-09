import os
import subprocess

def main():

    script_path = "/home/kemove/github/FlexGaussian/compress.py"
    model_path = "/home/kemove/data/3dgs-model/bicycle/"
    data_device = "cuda"
    output_path = "/home/kemove/efficient-output/"
    source_path = "/home/kemove/data/mipnerf360/bicycle/"  # 这里设置你的 source_path

    # 准备命令
    command = [
        "python", script_path,
        "-s", source_path,  
        "-m", model_path,
        "--output_path", output_path,
        "--eval"
    ]

    # 调用 compress.py 脚本
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing the script: {e}")

if __name__ == "__main__":
    main()


