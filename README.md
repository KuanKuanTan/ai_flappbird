# AI Flappy Bird CNN

這個專案把原本的 Flappy Bird Pygame 遊戲整理成一套可用來蒐集資料、訓練 CNN、載入模型自動遊玩，並支援人類對戰 AI 的平台。

目前這個版本是 **監督式學習 CNN**，不是強化學習。你可以手動玩遊戲，同時把每一幀影像與你的操作記錄下來，之後直接拿這些資料訓練模型。

## 功能

- 原版 Flappy Bird 遊戲
- 手動錄製訓練資料
- Expert policy 自動產生示範資料
- CNN 影像前處理
- CNN 監督式訓練
- TensorBoard 記錄
- Matplotlib 訓練曲線輸出
- AI 自動遊玩
- Human vs AI battle mode
- 模型評估

## 專案結構

```text
ai_flappybird/
├─ assets/                 # 遊戲圖片與音效
├─ config/
│  └─ flappy_ai.toml       # 集中設定檔
├─ data/
│  └─ datasets/            # 錄製或自動蒐集的資料
├─ models/                 # 最佳模型
├─ checkpoints/            # 訓練 checkpoint
├─ logs/                   # TensorBoard 與圖表輸出
├─ src/
│  ├─ ai/                  # 資料集、CNN、訓練、評估
│  ├─ env/                 # 環境封裝
│  ├─ game/                # 可重用遊戲核心
│  ├─ entities/            # 原始專案物件
│  └─ utils/               # 原始工具模組
├─ main.py                 # CLI 入口
└─ pyproject.toml          # 套件與依賴
```

## 環境需求

- Python 3.11 或 3.12
- Pygame
- NumPy
- OpenCV
- PyTorch
- TensorBoard
- Matplotlib

## 安裝

先進到專案目錄：

```bash
cd C:\Users\psy\Desktop\User\Kuan\Code\ai_flappybird
```

建議使用虛擬環境：

```bash
python -m venv .venv
.venv\Scripts\activate
```

安裝依賴：

```bash
pip install -e .
```

如果你只想快速安裝主要依賴，也可以：

```bash
pip install pygame numpy opencv-python torch tensorboard matplotlib
```

## 快速開始

### 1. 手動玩遊戲

```bash
python main.py play
```

操作：

- `SPACE` 或 `UP`: flap
- `ESC`: 離開
- 死掉後 `R`: 重新開始

### 2. 一邊玩一邊錄資料

這是最重要的模式。如果你想讓 AI 學你的操作，就用這個：

```bash
python main.py record --episodes 5
```

操作：

- `SPACE` 或 `UP`: flap
- 每局結束後 `R`: 儲存這一局並開始下一局
- `ESC`: 離開

每一局會存成一個 `.npz` 檔，預設放在：

```text
data/datasets/
```

每個檔案包含：

- `states`: CNN 輸入狀態，形狀為 `(N, 4, 84, 84)`
- `actions`: 對應動作，`0=WAIT`、`1=FLAP`
- `frames`: 原始 RGB 畫面
- `score`: 該局分數

### 3. 自動蒐集 expert 資料

如果你想先快速生一些訓練資料，可以用內建 heuristic expert：

```bash
python main.py collect --episodes 100 --controller expert
```

### 4. 訓練 CNN

```bash
python main.py train
```

如果你想從 checkpoint 繼續：

```bash
python main.py train --resume
```

### 5. 讓 AI 自動玩

```bash
python main.py ai-play --model models/flappy_cnn.pth
```

### 6. 人類 vs AI 對戰

```bash
python main.py battle --model models/flappy_cnn.pth
```

左邊是你，右邊是 AI。

### 7. 評估模型

```bash
python main.py evaluate --model models/flappy_cnn.pth --episodes 10
```

## CLI 指令總覽

```bash
python main.py --help
```

可用模式：

- `play`: 手動玩
- `record`: 手動玩並錄製圖片與動作
- `collect`: 自動蒐集資料
- `train`: 訓練 CNN
- `ai-play`: 載入模型自動玩
- `battle`: 人類對戰 AI
- `evaluate`: 評估模型

## CNN 輸入流程

CNN 直接吃遊戲畫面，不吃手工特徵。

處理流程：

1. 抓取目前遊戲畫面
2. 轉灰階
3. Resize 到 `84x84`
4. 將像素正規化到 `0~1`
5. 堆疊最近 `4` 幀

最終輸入形狀：

```text
(4, 84, 84)
```

## CNN 架構

模型在 `src/ai/model.py`。

結構如下：

1. `Conv2d(4, 32, kernel_size=8, stride=4)`
2. `ReLU`
3. `Conv2d(32, 64, kernel_size=4, stride=2)`
4. `ReLU`
5. `Conv2d(64, 64, kernel_size=3, stride=1)`
6. `ReLU`
7. `Flatten`
8. `Linear(3136, 512)`
9. `ReLU`
10. `Linear(512, 2)`

輸出：

- `0 = WAIT`
- `1 = FLAP`

## 訓練資料格式

每局資料會存成：

```text
episode_00000.npz
episode_00001.npz
...
```

你可以用 NumPy 讀：

```python
import numpy as np

data = np.load("data/datasets/episode_00000.npz")
print(data["states"].shape)
print(data["actions"].shape)
print(data["frames"].shape)
print(data["score"])
```

## 設定檔

集中設定檔：

[`config/flappy_ai.toml`](C:/Users/psy/Desktop/User/Kuan/Code/ai_flappybird/config/flappy_ai.toml:1)

可調整內容包含：

- 遊戲尺寸、FPS、重力、管道間距
- 前處理大小與 frame stack 數量
- batch size、learning rate、epochs
- logs/checkpoints/models 路徑
- deterministic reset
- expert policy 參數

常用欄位：

```toml
[preprocessing]
frame_width = 84
frame_height = 84
frame_stack = 4

[training]
batch_size = 64
learning_rate = 0.0005
epochs = 20
```

## 輸出目錄

- `data/datasets/`: 錄製資料與 expert 蒐集資料
- `models/flappy_cnn.pth`: 最佳模型
- `checkpoints/cnn_latest.pth`: 最新 checkpoint
- `logs/`: TensorBoard 與訓練圖

## 查看 TensorBoard

```bash
tensorboard --logdir logs
```

然後打開瀏覽器查看 TensorBoard 頁面。

## 常見工作流程

### 工作流程 A：用你自己的操作訓練

1. `python main.py record --episodes 20`
2. `python main.py train`
3. `python main.py ai-play --model models/flappy_cnn.pth`

### 工作流程 B：先用 expert 資料快速試跑

1. `python main.py collect --episodes 100 --controller expert`
2. `python main.py train`
3. `python main.py evaluate --model models/flappy_cnn.pth --episodes 10`

### 工作流程 C：你和 AI 對打

1. 先準備模型
2. `python main.py battle --model models/flappy_cnn.pth`

## 已知限制

- 目前是監督式 CNN，不是 DQN 或 PPO。
- 模型品質高度依賴你錄製的資料量與資料品質。
- 如果只錄很少局，AI 常常只會學到很短視的操作。
- `frames` 會占用較多磁碟空間；若只想訓練 CNN，真正必要的是 `states` 與 `actions`。

## Troubleshooting

### 1. OpenCV 和 NumPy 相容性錯誤

如果你看到這類錯誤：

```text
ImportError: numpy.core.multiarray failed to import
RuntimeError: module compiled against ABI version ...
```

代表 `opencv-python` 和 `numpy` 版本不相容。先執行：

```bash
pip uninstall -y opencv-python opencv-python-headless numpy
pip install --upgrade numpy opencv-python
```

如果還不行，再試：

```bash
pip install --force-reinstall "numpy<2" opencv-python
```

### 2. TensorBoard 無法啟動

如果訓練時出現 TensorBoard 未安裝，補裝即可：

```bash
pip install tensorboard
```

### 3. 沒有 CUDA，能不能跑

可以。程式會自動 fallback 到 CPU，只是訓練會比較慢。

### 4. 錄製資料後看不到檔案

檢查：

- 你是否真的在每局結束後按了 `R`
- `data/datasets/` 是否存在
- 目前工作目錄是不是專案根目錄

## 開發備註

- 主要入口：[main.py](C:/Users/psy/Desktop/User/Kuan/Code/ai_flappybird/main.py:1)
- CLI 與模式切換：[src/flappy.py](C:/Users/psy/Desktop/User/Kuan/Code/ai_flappybird/src/flappy.py:1)
- 遊戲核心：[src/game/world.py](C:/Users/psy/Desktop/User/Kuan/Code/ai_flappybird/src/game/world.py:56)
- 資料記錄：[src/ai/dataset.py](C:/Users/psy/Desktop/User/Kuan/Code/ai_flappybird/src/ai/dataset.py:59)
- CNN 模型：[src/ai/model.py](C:/Users/psy/Desktop/User/Kuan/Code/ai_flappybird/src/ai/model.py:7)
- 訓練流程：[src/ai/trainer.py](C:/Users/psy/Desktop/User/Kuan/Code/ai_flappybird/src/ai/trainer.py:71)

## 下一步建議

如果你要把這個專案往更完整的 AI 平台推，可以接著做：

- 自動切分 train/val dataset manifest
- 資料平衡分析
- 模型推論延遲量測
- 更好的 imitation learning 標註策略
- 後續再接強化學習版本
