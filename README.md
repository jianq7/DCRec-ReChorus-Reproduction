# DCRec-ReChorus: Debiased Contrastive Learning for Sequential Recommendation

本仓库是 WWW 2023 论文 **DCRec** (Debiased Contrastive Learning for Sequential Recommendation) 的复现代码，基于 **ReChorus** 框架实现。

## 🛠️ 环境依赖
- Python 3.10
- PyTorch 1.12.1
- numpy:1.22.3
- pandas:1.4.4
- scikit-learn:1.1.3
- scipy:1.7.3
- tqdm:4.66.1
- ipython:8.10.0
- jupyter:1.0.0
- PyYAML
- 也可直接运行以下命令安装依赖：
  ```bash
  pip install -r requirements.txt

## **数据集MovieLens-1M下载**

- 先从给出的网址下载原始数据zip文件（https://files.grouplens.org/datasets/movielens/ml-1m.zip）
- 将压缩文件解压放置在 `data/MovieLens_1M/` 目录下
- 使用 Jupyter Notebook 打开 `data/MovieLens_1M/our_MovieLens-1M.ipynb`
- 直接点击 **"Run All"**
- 运行结束后，`data/MovieLens_1M/` 目录下会出现 `train.csv`, `dev.csv`, `test.csv`，即可开始训练。

## 🚀 运行指南

### 运行复现的DCRec

- 在pycharm或VS Code终端运行下面的命令：

  ```bash
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  ```

### **不同模型在 Grocery 和 MovieLens-1M 数据集上的性能对比**

- **在Grocery数据集上**

  ```bash
  # DCRec
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  # SASRec
  python src/main.py --model_name SASRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --num_workers 0 --epoch 50
  # GRU4Rec
  python src/main.py --model_name GRU4Rec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --num_workers 0 --epoch 50
  ```

  

- **在MovieLens-1M数据集上**

  ```bash
  # DCRec
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  # SASRec
  python src/main.py --model_name SASRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1
  # GRU4Rec
  python src/main.py --model_name GRU4Rec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1
  ```

  

### 消融实验

**每次做下一个实验前先把前一次实验修改过的代码恢复为最初版本**

- **完整的DCRec**

  ```bash
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  ```

  

- **移除对比学习**

  ```bash
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0 --aug_prob 0.2 --temperature 1.0
  ```

  

- **移除“保留末位”约束（w/o Safe Masking）**

  ```bash
  # 首先将DCRec.py文件中的 _augment_safe_mask方法替换成下面的代码
  def _augment_safe_mask(self, seqs, lengths):
      """消融安全增强：随机mask所有有效物品（包括最后一个交互项）"""
      aug_seqs = seqs.clone()
      batch_size, max_len = seqs.shape
      # 移除「非最后位置」限制，仅保留随机概率+有效物品筛选
      rand_matrix = torch.rand(aug_seqs.shape, device=self.device)
      final_mask = (rand_matrix < self.aug_prob) & (aug_seqs > 0)
      # 执行mask（置0）
      aug_seqs.masked_fill_(final_mask, 0)
      return aug_seqs
  ```

  ```bash
  # 然后运行如下命令
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  ```

  

- **去除Projector投影层**

  -  注释DCRec.py文件__init__中如下代码

    ```bash
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.aug_prob = args.aug_prob
        self.cl_weight = args.cl_weight
        self.tau = args.temperature
    	self.projector = None
    	
        # ========== 注释以下Projector相关代码 ==========
        # if self.cl_weight > 0:
        #     self.projector = nn.Sequential(
        #         nn.Linear(self.emb_size, self.emb_size),
        #         nn.ELU(),
        #         nn.Linear(self.emb_size, self.emb_size)
        #     )
        #     self.apply(self.init_weights)
    
    ```

    

  -   修改 calculate_loss和calculate_constrastive_loss 中的代码

    ```bash
    # calculate_loss中
    # 原代码
    if self.training and self.cl_weight > 0 and self.projector is not None:
    # ========== 修改后 ==========
    if self.training and self.cl_weight > 0:
    # 原代码
    # z1 = self.projector(seq_emb1)
    # z2 = self.projector(seq_emb2)
    # ========== 替换为以下代码 ==========
    z1 = seq_emb1  # 直接使用序列嵌入，无投影
    z2 = seq_emb2
    
    # calculate_contrastive_loss
    # 原代码
    if not self.training or self.cl_weight <= 0 or self.projector is None:
    # ========== 修改后 ==========
    if not self.training or self.cl_weight <= 0:
    # 原代码
    # z1 = self.projector(seq_emb1)
    # z2 = self.projector(seq_emb2)
    # ========== 替换为以下代码 ==========
    z1 = seq_emb1  # 直接使用序列嵌入，无投影
    z2 = seq_emb2
    ```

    

  - 运行命令

    ```bash
    python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
    ```





### 超参实验

- **cl_weight**

  ```bash
  # 1. cl_weight=0.02
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.02 --aug_prob 0.2 --temperature 1.0
  
  # 2. cl_weight=0.05
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  
  # 3. cl_weight=0.08
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.08 --aug_prob 0.2 --temperature 1.0
  
  # 4. cl_weight=0.12
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.12 --aug_prob 0.2 --temperature 1.0
  ```

- **aug_prob**

  ```bash
  # 1. aug_prob=0.1
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.1 --temperature 1.0
  
  # 2. aug_prob=0.2
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  
  # 3. aug_prob=0.25
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.25 --temperature 1.0
  
  # 4. aug_prob=0.35
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.35 --temperature 1.0
  ```

- **temperature**

  ```bash
  # 1. temperature=1.0
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 1.0
  
  # 2. temperature=2.0
  python src/main.py --model_name DCRec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MovieLens_1M --num_workers 0 --epoch 50 --test_all 1 --cl_weight 0.05 --aug_prob 0.2 --temperature 2.0
  ```

  