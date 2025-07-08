# 基于变分模态分解的高速列车轨道波磨检测模型

> 项目blog:

**项目单位：苏州大学 | 项目编号：202310285155Y**

## 1. 项目简介

随着我国高速铁路的飞速发展，列车运行速度不断提升，钢轨表面在轮轨高频相互作用下产生的波浪形磨耗（简称“波磨”）问题日益凸出。波磨不仅会引发剧烈的轮轨振动和噪声，影响乘客舒适度，更会加速车辆和轨道部件的疲劳损伤，对行车安全构成严重威胁。因此，建立一套高效、精准的轨道波磨检测模型具有至关重要的现实意义。

本项目针对传统信号处理方法在波磨检测中的局限性，提出了一种**基于自适应变分模态分解（Adaptive Variational Mode Decomposition, AVMD）的轨道波磨检测模型**。该模型通过优化VMD算法，克服了其需要预设分解模态数K值的难题，实现了对复杂振动信号的自适应、高精度分解。



## 2. ⚙️ 环境配置与运行

### 2.1  本项目的所有依赖项都记录在 `requirements.txt` 文件中。激活虚拟环境后，运行以下命令一键安装：

    ```bash
    pip install -r requirements.txt
    ```
    该文件将自动安装以下核心库的指定版本：
    * `matplotlib==3.7.2`
    * `numpy==1.24.1`
    * `pandas==2.0.3`
    * `Pillow==11.3.0`
    * `scipy==1.16.0`
    * `openpyxl`

    **注意**: 项目所需的 `vmdpy.py` 等自定义模块已包含在 `Code/` 目录中，无需额外安装。
    
### 2.2 🚀 执行流程

本项目的执行遵循“数据预处理 -> 特征提取 -> GUI集成应用 -> 阈值计算 -> 可视化验证”的流程。

#### **第1步：提取波磨区段特征**
`vmd_project`
1. 对波摩区段的信号可视化,得到原始信号时域图，经过快速傅里叶变换（FFT）得到原始信号频域图
2. 对原始信号进行自适应变分模态分解（AVMD），得到特定数量的本征模态函数（IMF）
3. 可视化分解得到的IMF，得到时域图与频域图

对正常区段采用同样的操作，使用多个评测指标对比正常区段与波摩区段的IMF，最终选取分解得到的IMF1的均方根（RMS）与标准差（STD）的平均值之和作为波摩检测阈值


#### **第2步：计算波磨检测阈值**

利用**正常区段**数据计算基准阈值。

* **对于AVMD方法**:
    1.  修改 `IMF_time_rms_std_multiple_left.py` 和 `IMF_time_rms_std_multiple_right.py` 中的 `file_pattern`，指向 `Dataset/正常区段/`。
    2.  运行脚本，生成 `normal_result_left_output.xlsx` 和 `normal_result_right_output.xlsx`。
    3.  在控制台记录下计算出的最终阈值。
    4.  对**波磨区段**数据进行同样操作。
       * 修改相应脚本中的 `file_pattern`，使其指向 `Dataset/波磨区段/`，然后运行，生成 `bomo_result_..._output.xlsx` 系列文件。

* **对于VMD方法 (固定K值)**:
    1.  修改 `IMF_time_rms_std_multiple_left_vmd.py` 和 `IMF_time_rms_std_multiple_right_vmd.py` 中的 `file_pattern`，指向 `Dataset/正常区段/`。
    2.  运行脚本，生成 `normal_result_left_fixed_k_output.xlsx` 等文件。
    3.  记录控制台输出的阈值。
    4.  对**波磨区段**数据进行同样操作。
      * 修改相应脚本中的 `file_pattern`，使其指向 `Dataset/波磨区段/`，然后运行，生成 `bomo_result_..._output.xlsx` 系列文件。



#### **第3步：波摩检测可视化与验证**

1.  打开 `波磨阈值检查v1连线(AVMD).py` 或 `波磨阈值检查v1连线(VMD).py`。
2.  确保文件路径指向第1、2步生成的 `.xlsx` 结果文件。
3.  将第1步记录下的对应方法的**阈值**填入脚本中的 `check` 变量。
4.  运行脚本，将显示一张包含正常区段（蓝线）、波磨区段（红线）和检测阈值（虚线）的对比图，并输出最终检测率。

**AMVD 结果:**
![AVMD](https://github.com/user-attachments/assets/1137f60a-6944-4a26-9f14-648f71e58b7e)
**VMD 结果:**
![VMD](https://github.com/user-attachments/assets/d5cae83c-2e43-4506-852a-ff59c6075716)



## 3. 📖 数据集说明

* `Dataset/原始数据`: 包含从检测设备直接导出的`.csv`文件和对波磨区段位置进行说明的`.xls`文件。
* `Dataset/正常区段`, `波磨区段`, `混合数据`: 这些是经过预处理、格式统一为`.xlsx`的数据。每个文件包含 `Left` 和 `Right` 两列，分别代表左右轨的振动加速度信号。

## 4. 🤝 贡献与许可

本项目采用 **GNU General Public License v3.0 (GPLv3)** 开源许可证。详情请见 `LICENSE` 文件。
