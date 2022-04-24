# 运行指南

`ipynb` 格式文件为 `jupyter notebook` 格式文件，本实验的 `notebook` 代码使用 `ipython` 内核。

在 `anaconda` 完整安装的的 `base` 环境下应该可以运行，但未经验证。若出现无法运行的情况可通过以下方式复现我的工作环境。

## TL;DR

```
conda env create -f my_env.yml
conda activate cm_my_env
jupyter notebook
```

## env create

在命令行中：
```
conda env create -f my_env.yml
```
若 conda 不在环境变量中可使用 `anaconda prompt` 进入当前路径，即 `xxx/src` 下执行。

创建虚拟环境成功后执行 `conda activate cm_my_env` 启动虚拟环境。

若创建过程出错可尝试修改 conda 配置文件 `.condarc`，添加阿里镜像源。
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - http://mirrors.aliyun.com/anaconda/pkgs/main
  - http://mirrors.aliyun.com/anaconda/pkgs/r
  - http://mirrors.aliyun.com/anaconda/pkgs/msys2
custom_channels:
  conda-forge: http://mirrors.aliyun.com/anaconda/cloud
  msys2: http://mirrors.aliyun.com/anaconda/cloud
  bioconda: http://mirrors.aliyun.com/anaconda/cloud
  menpo: http://mirrors.aliyun.com/anaconda/cloud
  pytorch: http://mirrors.aliyun.com/anaconda/cloud
  simpleitk: http://mirrors.aliyun.com/anaconda/cloud
```
而后再次尝试创建

## 启动jupyter notebook

在 `cm_my_env` 启动的状态下，在当前工作路径 `xxx/src` 下执行 `jupyter notebook` 启动 jupyter notebook。点击 `Lab*.ipynb` 即可进入相应 `notebook`。

> 若电脑中有 `PyCharm` 等 IDE 可直接在 IDE 下打开当前目录

在运行 `jupyter notebook` 之前将笔记本设置为可信，点击 `run all` 按钮或者在 Cell -> Run All 运行笔记本

## 说明

本实验代码在提交前已在其它机器上复现，若代码执行出错或无法运行望请老师或助教联系我。
