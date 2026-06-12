# AI Flappy Bird CNN 專案完整介紹

這個專案是一套把經典遊戲 Flappy Bird 轉換成 AI 訓練平台的系統。它不只是把遊戲做出來，而是把「遊戲本體」、「資料錄製」、「影像前處理」、「CNN 訓練」、「AI 推論」、「成果展示」全部串成一條完整流程。整個專案的核心目標，是讓電腦透過觀察遊戲畫面與玩家示範操作，學會控制 Flappy Bird 裡的小鳥飛行，並在之後用訓練好的模型自動遊玩，甚至與真人即時對戰。

這套系統採用的是影像式監督學習。也就是說，AI 不是直接讀取手工整理好的座標特徵，而是像人一樣先看到畫面，再根據畫面判斷現在應該拍翅還是等待。這個設計讓專案同時具備遊戲開發、電腦視覺、深度學習、資料工程與系統整合的價值。

目前這個版本使用的是 CNN 模型，學習方式是 imitation learning，也就是模仿學習。系統先讓玩家手動玩遊戲，再把玩家當下看到的畫面和玩家的決策記錄下來，最後交給神經網路學習「在這種畫面下，人通常會怎麼操作」。訓練完成後，模型就能在真正的遊戲畫面上做即時推論，變成可執行的遊戲 AI。

---

## 這個專案要解決什麼問題

一般的 Flappy Bird 遊戲只適合人玩，遊戲程式本身通常不會考慮資料蒐集、模型訓練、實驗重現或 AI 推論。因此如果想把遊戲變成 AI 專題，就必須先解決幾個問題。

第一，如何把遊戲畫面轉成機器學習可使用的輸入。原始畫面是 RGB 影像，大小也不是為模型設計的，因此不能直接拿來訓練。

第二，如何在玩家遊玩時同步錄下畫面與動作，而且不要把明顯錯誤的死亡尾段一起存進資料集。

第三，如何把零散的遊戲操作整理成可訓練的資料格式，並且能持續擴充、檢查與管理。

第四，如何讓訓練好的模型真正接回遊戲，在每一幀畫面下做出控制決策。

第五，如何讓整個系統不只是 demo，而是有清楚架構、能保存模型、能顯示訓練曲線、能做 AI 展示與人機對戰的完整平台。

這個專案的本質，就是在解決上述所有問題，讓一個原本只是單純互動遊戲的程式，變成一套可訓練、可驗證、可展示的 AI 系統。

---

## 這個系統在做什麼

最簡單的理解方式是：**先讓玩家玩給電腦看，再讓電腦學著玩。**

具體流程如下。

玩家先手動玩 Flappy Bird。系統在背景中同步記錄每一幀遊戲畫面，以及玩家當下是否有按下拍翅。當錄到足夠資料之後，系統會把這些畫面做影像前處理，轉成固定大小、固定格式的神經網路輸入。接著利用這些輸入和玩家動作標籤訓練一個 CNN 分類模型，讓模型學會在特定畫面狀態下判斷應該選擇 `WAIT` 還是 `FLAP`。最後把訓練好的模型載入回遊戲，讓 AI 直接根據畫面做決策，自動控制角色飛行。

這個專案的重點不只是「有一個 AI」，而是整條資料流都完整打通：

- 可以錄資料
- 可以清理資料
- 可以檢查資料
- 可以訓練模型
- 可以儲存模型
- 可以用模型自動玩
- 可以與真人對戰

因此，它不是只有遊戲，也不是只有模型，而是一個小型 AI 訓練平台。

---

## 系統完整架構

從系統設計角度來看，這個專案可以分成七層。

### 1. 入口層

入口層負責接收使用者命令並決定系統要進入哪一種模式。使用者不需要直接呼叫內部模組，只要透過命令列，就能進入手動遊玩、錄製資料、訓練模型、AI 自動遊玩、資料檢查或人機對戰等功能。

### 2. 設定層

設定層把遊戲參數、影像前處理參數、資料錄製規則、訓練超參數、battle 設定全部集中管理。這讓程式裡不需要到處出現硬編碼數字，並且可以快速修改實驗條件。

### 3. 遊戲核心層

這一層是真正驅動 Flappy Bird 的地方。它負責小鳥、管線、地板、分數、碰撞判定與畫面渲染。這是整個專案的地基，因為不管操作者是人、AI、還是 expert policy，最後都必須使用這一層來更新世界狀態。

### 4. 環境與前處理層

這一層把原本偏向人類遊玩的遊戲，轉換成 AI 可以理解的形式。環境層負責每一步的狀態更新與回傳，前處理層負責把畫面轉成 CNN 的輸入格式。

### 5. 資料集層

資料集層負責錄製、整理、保存與讀取資料。這裡不只是單純存檔，還包含死亡尾段裁切、session 串接、資料檢查與資料格式標準化。

### 6. 模型與訓練層

這一層負責定義 CNN、讀取資料、執行訓練、保存最佳模型與 checkpoint，並輸出 TensorBoard 紀錄與訓練曲線。

### 7. 展示與評估層

這一層負責載入模型做推論，並透過 AI 自動遊玩、人機對戰與資料 inspect 等功能，把結果以可理解的方式呈現出來。

如果把整個資料流寫成一條線，就是：

**玩家操作或 expert 策略 -> 遊戲核心更新 -> 畫面擷取 -> 前處理 -> 錄製成資料集 -> CNN 訓練 -> 載入模型 -> 即時推論 -> AI 控制遊戲**

---

## 專案目錄與模組分工

這個專案不是所有邏輯都塞在一支主程式裡，而是透過目錄結構做明確分層。

### 根目錄

- `main.py`
  - 專案命令列入口。使用者執行程式時會先進入這裡，再轉交給內部主流程。

- `config/`
  - 存放設定檔，例如 `flappy_ai.toml`。這裡集中管理遊戲參數、影像前處理與訓練超參數。

- `assets/`
  - 存放遊戲素材，例如背景、小鳥、地板、柱子與音效。

- `data/datasets/`
  - 存放錄製好的資料集，例如 `session_00000.npz`。

- `models/`
  - 存放最佳模型權重，例如 `flappy_cnn.pth`。

- `checkpoints/`
  - 存放訓練中的 checkpoint。

- `logs/`
  - 存放 TensorBoard 紀錄與訓練圖表。

### `src/` 底下的主要模組

- `src/flappy.py`
  - 整個系統的主控制器，負責切換所有模式。

- `src/game/`
  - 遊戲核心、設定與資產管理。

- `src/env/`
  - 將遊戲封裝成環境介面，供資料蒐集與模型推論共用。

- `src/ai/`
  - 前處理、資料集、CNN 模型、訓練、評估與視覺化。

- `src/entities/`
  - 原始版本的遊戲物件模組，作為基礎與相容保留。

- `src/utils/`
  - 工具函式與常數。

這種拆法的意義在於責任清楚。每個資料夾都處理一種層級的問題，而不是把畫面、模型、資料錄製、訓練全部混在一起。

---

## 使用者可以怎麼操作這套系統

這個專案提供多種模式，每種模式都對應不同的用途。

### `play`

純手動遊玩模式。這個模式只負責遊戲本身，不錄資料、不做 AI 推論。適合測試遊戲是否正常。

### `record`

手動錄製模式。玩家一邊玩，系統一邊記錄：

- 原始畫面
- 前處理後的 CNN 狀態
- 玩家動作

每一局結束後，系統會自動刪除死亡前最後幾幀，再把可用片段接到同一個 session 裡。這是產生訓練資料的主要方式。

### `collect`

自動蒐集模式。這個模式不需要真人玩，而是用內建的 heuristic expert policy 自動控制小鳥，快速產生示範資料。

### `train`

訓練模式。系統會讀取資料集、切分訓練集與驗證集、建立 DataLoader、訓練 CNN，並儲存最佳模型與 checkpoint。

### `ai-play`

AI 自動遊玩模式。系統會載入模型，根據每一幀畫面即時推論，控制小鳥飛行。

### `battle`

人機對戰模式。左邊是人，右邊是 AI，雙方在同樣規則下比賽，可以直觀比較得分、生存時間與距離。

### `inspect`

資料檢查模式。可查看某一份錄製檔的原始畫面與 CNN 輸入 stack，確認資料是否合理。

---

## 資料是怎麼被錄製的

資料錄製是這個專案最重要的環節之一，因為模型最後學到什麼，取決於你餵給它的資料。

當玩家使用 `record` 模式時，系統會在每一幀執行以下工作：

1. 讀取目前遊戲畫面
2. 讀取玩家是否按下拍翅鍵
3. 把當前畫面保存成原始 RGB frame
4. 將畫面前處理成 CNN 狀態
5. 將動作記成標籤
6. 繼續推進遊戲到下一幀

當這一局結束時，系統不會直接原封不動存起來，而是先進行資料清理。若小鳥是因為撞到柱子或其他死亡條件而失敗，系統會自動刪掉死亡前最後幾幀。這樣做是為了避免把明顯失敗且無法挽回的錯誤狀態一併學進模型裡。

清理完成後，這一段可用資料不會單獨拆成一堆碎片，而是會被串接到同一個 recording session 內。這讓資料管理更整齊，也讓整個資料流更接近連續示範。

---

## 為什麼要刪掉死亡前幾幀

這個設計非常關鍵。

在 Flappy Bird 裡，死亡前最後幾幀常常是已經來不及修正的狀態。從人類角度看，那些畫面通常代表的是「這局已經失敗了」，而不是一個值得模仿的決策過程。如果把這些幀直接留在資料集中，模型很可能會學到一些在邏輯上無法挽回的操作模式。

例如，小鳥已經貼近柱子或地板，玩家雖然在最後一刻按了拍翅，但實際上已經救不回來。如果模型把這種情況也視為正常示範，它之後可能會在危險狀態下過度延後決策，學到不穩定的行為。

因此，刪除死亡前最後幾幀的目的，是提升資料品質，減少失敗尾段污染。這不是單純刪資料，而是在幫模型保留更有學習價值的決策區段。

---

## 影像前處理在做什麼

模型不能直接吃原始遊戲畫面，因為畫面太大、資訊太多，而且彩色資訊對這個任務不一定必要。因此需要先做前處理。

目前的前處理流程如下：

1. 從遊戲畫面擷取 RGB 影像
2. 轉成灰階
3. Resize 到 `84 x 84`
4. 將像素值正規化到 `0 ~ 1`
5. 堆疊最近 `4` 張影格

最後得到的輸入形狀是：

```text
(4, 84, 84)
```

這個設計的目的，不只是讓輸入變小、變統一，更重要的是讓模型看到時間變化。因為 Flappy Bird 是動態遊戲，模型若只看單張圖片，很難知道小鳥是往上還是往下。連續 4 幀的設計能提供足夠的運動資訊。

---

## CNN 模型到底在學什麼

模型的表面任務是分類，也就是判斷當前應該選擇 `WAIT` 還是 `FLAP`。但本質上，它是在學習一個從畫面到動作的映射關係。

當模型看到畫面時，它會試著從中辨識：

- 小鳥在畫面中的相對位置
- 小鳥和下一根柱子的距離
- 柱子 gap 的中心位置
- 小鳥當下是上升還是下降
- 是否接近地板或障礙

模型本身不會用文字理解這些概念，但透過卷積層會逐步學到這些空間特徵。最後它輸出兩個值，用來表示現在比較像是應該等待還是拍翅。

因為這是模仿學習，所以模型學到的其實是玩家或 expert policy 的操作風格。如果錄製資料偏保守，模型也可能偏保守；如果錄製資料拍翅很頻繁，模型也可能學得更積極。

---

## 訓練流程是怎麼運作的

訓練流程可以分成幾個步驟。

首先，系統會從資料集目錄中讀取 `session_*.npz` 或 `episode_*.npz`。接著用 `EpisodeDataset` 把資料展成單筆樣本，再用 PyTorch 的 DataLoader 做批次讀取。

之後系統會：

- 建立 CNN 模型
- 選擇 CPU 或 GPU
- 設定 optimizer 與 loss function
- 切分訓練集與驗證集
- 逐個 epoch 執行訓練與驗證
- 記錄 train loss、val loss、train accuracy、val accuracy
- 保存最佳模型
- 定期保存 checkpoint
- 輸出 TensorBoard 與 Matplotlib 曲線

這代表整個訓練過程不是單純跑一次，而是有完整的實驗管理能力。你可以追蹤訓練變化、可以中斷後續跑，也能比較不同參數下的結果。

---

## 訓練完成後，AI 怎麼控制遊戲

當模型訓練完成後，系統會把它載入回遊戲裡。

AI 遊玩時，每一幀都會執行相同流程：

1. 擷取當前遊戲畫面
2. 做影像前處理
3. 更新 frame stack
4. 把 state 丟進 CNN
5. 取得兩個輸出值
6. 選擇較高的類別作為實際動作
7. 把這個動作交給遊戲核心

於是模型就不只是靜態分類器，而是真正變成控制器，持續影響遊戲世界的下一步。

---

## Human vs AI 對戰模式有什麼意義

如果只有 AI 自動玩，觀眾很難直觀理解 AI 的強弱。因此專案加入了 battle 模式，讓人類與 AI 同場對戰。

這個模式的好處有三個：

第一，展示效果非常直觀。觀眾可以同時看到人和 AI 的操作結果。

第二，可以形成比較基準。雙方在相同規則下遊玩，可以比較得分、生存時間與移動距離。

第三，可以快速驗證模型是否真的有實用價值。如果 AI 能穩定與人類競爭，就代表資料錄製與訓練流程具有實際效果。

---

## 為什麼這不只是遊戲，而是一個 AI 平台

這個專案之所以有價值，不是因為它重新做了一個 Flappy Bird，而是因為它把一個遊戲完整轉換成 AI 系統。

它具備機器學習專案最關鍵的完整鏈條：

- 有資料來源
- 有資料清理
- 有前處理
- 有資料檢查
- 有模型定義
- 有訓練流程
- 有模型保存
- 有推論流程
- 有展示與評估方式

因此，它比一般單純的遊戲專案更接近真實 AI 專案，也比一般單純的模型 demo 更接近完整應用系統。

---

## 重點程式碼解說

下面這一段不是逐行解釋語法，而是從系統責任角度說明：每個核心檔案在整個平台中扮演什麼角色。

### `main.py`

這是專案入口。使用者執行 `python main.py ...` 時，會從這裡進入，再轉呼叫內部主流程。它讓使用方式簡單固定，不需要使用者理解內部模組結構。

### `src/flappy.py`

這是整個專案的主控制器，也是流程調度中心。它負責：

- 解析命令列參數
- 初始化 Pygame
- 載入設定與素材
- 根據模式切換到 `play`、`record`、`collect`、`train`、`ai-play`、`battle`、`inspect`

這支程式不是在做模型訓練本身，而是在協調整個系統何時該錄資料、何時該訓練、何時該推論。

### `src/game/config.py`

這支程式把整個專案的設定集中整理成結構化資料。它負責從 TOML 檔讀取：

- 遊戲參數
- 前處理參數
- 訓練超參數
- expert policy 參數
- battle 參數

它的價值在於讓所有實驗條件都可以從設定檔調整，而不是散落在程式碼各處。

### `src/game/assets.py`

這裡負責載入遊戲圖片與音效，並包裝成 `GameAssets`。背景、小鳥、上下柱子、地板、分數圖樣與音效都在這裡統一準備好，供遊戲核心使用。

### `src/game/world.py`

這是整個專案最關鍵的檔案，也是遊戲核心的核心。它定義了：

- `BirdState`
- `PipeState`
- `GameWorldInfo`
- `GameWorld`

`GameWorld` 的責任包括：

- 重置遊戲
- 初始化小鳥與管線
- 推進小鳥物理狀態
- 推進管線與地板
- 計算分數
- 判斷碰撞
- 渲染畫面
- 取得 RGB frame

最關鍵的方法是 `step(action)`。這個方法接收一個動作，然後完整推進世界一幀，最後回傳目前分數、是否死亡、碰撞類型、生存時間與距離。幾乎所有模式都依賴這個方法，因此它是遊戲核心與 AI 流程共同的樞紐。

另一個重要方法是 `get_expert_action()`。這是一個內建 heuristic policy，根據下一根柱子的 gap 與小鳥位置，決定要不要拍翅，用來產生自動示範資料。

### `src/env/flappy_env.py`

這一層把 `GameWorld` 和 `FrameProcessor` 結合起來，形成可被資料錄製與 AI 推論共用的環境封裝。

`reset()` 會重置世界並建立初始 state。

`step(action)` 會：

- 把動作交給 `GameWorld`
- 更新畫面
- 擷取 RGB frame
- 更新前處理 stack
- 回傳新的 state、reward、done 與 info

這個設計讓遊戲流程更像標準 AI 環境，也為未來擴充成強化學習保留了空間。

### `src/ai/processor.py`

這是前處理核心。`FrameProcessor` 負責：

- RGB 轉灰階
- Resize 成 84x84
- 正規化到 0 到 1
- 維護最近 4 幀的 stack

`preprocess()` 處理單幀。

`reset()` 用於建立初始 stack。

`push()` 用於追加新幀。

`state` 會回傳最終的 `(4, 84, 84)` NumPy 陣列。

這個模組等於定義了模型眼中的世界。模型看到什麼，完全由這裡決定。

### `src/ai/dataset.py`

這是資料工程核心，負責：

- 建立 PyTorch Dataset
- 自動蒐集 expert 資料
- 裁切死亡尾段
- 儲存連續 session
- 檢視資料內容

`EpisodeDataset` 負責在訓練時讀取 `.npz` 資料檔並展開成單筆樣本。

`trim_episode_tail()` 是資料清理的關鍵，負責刪除死亡前幾幀，減少失敗尾段污染。

`save_recording_session()` 會把多局保留下來的可用片段接成同一個 session。

`inspect_episode_record()` 則讓使用者能直接看到原始畫面與 CNN state stack，是資料驗證的重要工具。

### `src/ai/model.py`

這裡定義 CNN 模型 `FlappyCNN`。模型使用三層卷積與兩層全連接，輸出兩類動作：

- `WAIT`
- `FLAP`

它本身程式碼不長，但代表整個專案的學習核心。所有資料錄製與前處理，最終都是為了產生它可有效學習的輸入。

### `src/ai/trainer.py`

這裡是訓練流程主軸。它負責：

- 載入資料集
- 切分 train / val
- 建立 DataLoader
- 初始化模型、optimizer、loss function
- 執行 epoch 訓練
- 記錄 train/val loss 與 accuracy
- 儲存最佳模型
- 儲存 checkpoint
- 寫入 TensorBoard
- 更新訓練曲線

其中 `_run_epoch()` 負責執行單次 train 或 val，`train_model()` 則負責控制整體訓練流程。

### `src/ai/evaluate.py`

這裡負責模型載入與遊戲內評估。它會把模型放回遊戲環境中實際跑多局，統計：

- 平均分數
- 最佳分數
- 平均生存時間
- 平均距離

這代表評估不是只看分類指標，而是看模型在真實遊戲中的整體表現。

### `src/ai/visualization.py`

這支程式負責用 Matplotlib 即時更新訓練曲線，將 loss 與 accuracy 圖表保存到 logs 目錄。這讓訓練狀態變得直觀可視，而不是只能看終端輸出。

---

## 代表性程式碼片段

下面這些片段不是要把整份程式碼全部貼出來，而是挑出最有代表性的部分，讓讀者即使不打開專案，也能直接理解系統是怎麼運作的。

### 1. 遊戲核心每一幀如何更新

這段程式碼來自 `src/game/world.py` 的 `step(action)`。它代表整個遊戲核心的更新流程：先接收動作，再推進小鳥、管線與地板，最後檢查是否得分或死亡。

```python
def step(self, action: int) -> GameWorldInfo:
    if self.done:
        return self.info(False)

    if action == 1:
        self.flap()

    self._advance_bird()
    self._advance_pipes()
    self._advance_floor()
    self._spawn_pipe_if_needed()
    self._remove_old_pipes()
    passed_pipe = self._check_pipe_pass()

    if self._collides_with_floor() or self._collides_with_pipe():
        self.done = True
        self.assets.hit_sound.play()
        self.assets.die_sound.play()

    self.distance_traveled += self.config.game.pipe_speed
    self.survival_time += 1.0 / self.config.game.fps
    return self.info(passed_pipe)
```

這段的重點是，整個專案不管是手動玩、錄資料、expert 蒐集還是 AI 自動玩，最後都會走到這個更新流程。換句話說，這裡是所有模式共用的世界規則。

### 2. 畫面如何轉成 CNN 輸入

這段來自 `src/ai/processor.py` 的 `preprocess(frame)`。它展示了影像前處理的核心步驟。

```python
def preprocess(self, frame: np.ndarray) -> np.ndarray:
    gray = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2GRAY)
    resized = self.cv2.resize(
        gray,
        (self.settings.frame_width, self.settings.frame_height),
        interpolation=self.cv2.INTER_AREA,
    )
    return resized.astype(np.float32) / 255.0
```

這裡做的事情很直接：

- 把 RGB 影像轉灰階
- 縮放到固定大小 `84 x 84`
- 正規化到 `0 ~ 1`

也就是說，模型不是直接吃原始遊戲畫面，而是吃經過整理、標準化後的灰階影像。

### 3. 為什麼模型看到的是連續 4 幀

這段來自同一支檔案的 `state` 屬性與 `push(frame)`。它說明 frame stack 是如何形成的。

```python
def push(self, frame: np.ndarray) -> np.ndarray:
    processed = self.preprocess(frame)
    if not self.frames:
        return self.reset(frame)
    self.frames.append(processed)
    return self.state

@property
def state(self) -> np.ndarray:
    if not self.frames:
        raise RuntimeError("Frame stack is empty. Call reset() first.")
    return np.stack(self.frames, axis=0).astype(np.float32)
```

這表示系統不只看一張圖，而是把最近幾張圖疊在一起。這樣模型才能知道小鳥是往上飛還是往下掉，而不只是看到靜態位置。

### 4. 死亡前尾段如何被裁掉

這段來自 `src/ai/dataset.py` 的 `trim_episode_tail(...)`。它是資料品質控制的代表性程式碼。

```python
def trim_episode_tail(
    states: list[np.ndarray],
    actions: list[int],
    frames: list[np.ndarray],
    trim_frames: int,
    min_frames: int,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], int]:
    if trim_frames <= 0:
        return states, actions, frames, 0

    trimmed = min(trim_frames, max(0, len(states) - min_frames))
    if trimmed <= 0:
        return states, actions, frames, 0

    keep = len(states) - trimmed
    return states[:keep], actions[:keep], frames[:keep], trimmed
```

這段的重點不是語法，而是設計概念：系統允許刪掉死亡前最後幾幀，但又保證不會把整段資料刪到太短。這樣可以降低失敗尾段對模型的負面影響。

### 5. 多局資料如何串成同一個 session

這段來自 `src/ai/dataset.py` 的 `save_recording_session(...)`。它代表手動錄製資料不再是單局碎片，而是會被整理成一個可管理的 session。

```python
def save_recording_session(
    dataset_dir: str | Path,
    session_index: int,
    states: list[np.ndarray],
    actions: list[int],
    frames: list[np.ndarray],
    segment_lengths: list[int],
    segment_scores: list[int],
) -> Path:
    output_dir = Path(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"session_{session_index:05d}.npz"
    np.savez_compressed(
        file_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        frames=np.asarray(frames, dtype=np.uint8),
        segment_lengths=np.asarray(segment_lengths, dtype=np.int64),
        segment_scores=np.asarray(segment_scores, dtype=np.int64),
    )
    return file_path
```

這樣的資料結構讓訓練時比較好管理，也保留了每段原本的長度與分數資訊，方便後續分析。

### 6. CNN 模型本身長什麼樣子

這段來自 `src/ai/model.py`，是整個專案最核心的學習模型。

```python
class FlappyCNN(nn.Module):
    def __init__(self, input_channels: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
```

這裡可以清楚看到模型由三層卷積與兩層全連接組成，輸出只有兩類，分別是 `WAIT` 與 `FLAP`。

### 7. 訓練時每個 batch 在做什麼

這段來自 `src/ai/trainer.py` 的 `_run_epoch(...)`，它代表了模型在訓練時的核心步驟。

```python
for states, actions in loader:
    states = states.to(device)
    actions = actions.to(device)

    if training:
        optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        enabled=mixed_precision and device.type == "cuda",
    ):
        logits = model(states)
        loss = criterion(logits, actions)

    if training:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

這段顯示出訓練的基本機制：

- 從資料集中拿出一批 state 與 action
- 把 state 丟進模型得到 logits
- 計算 loss
- 反向傳播更新模型權重

也就是說，整個學習過程本質上是在反覆修正模型，讓它對畫面做出的動作判斷越來越接近示範資料。

### 8. 環境如何把遊戲與前處理接在一起

這段來自 `src/env/flappy_env.py` 的 `step(action)`。它展示了遊戲更新、畫面擷取與 state 生成是怎麼被包成同一個介面的。

```python
def step(self, action: int) -> tuple[object, float, bool, object]:
    info = self.world.step(action)
    self.world.render()
    frame = self.world.get_frame_rgb()
    self.processor.push(frame)
    reward = float(info.score)
    return self.get_state(), reward, info.done, info
```

這段的意義是：外部模組不需要自己去管遊戲怎麼畫、畫面怎麼抓、影像怎麼堆疊，它只要呼叫 `env.step(action)`，就能同時得到新的 state 與遊戲回饋。

---

## 幾個最關鍵的方法

如果只挑出最重要的方法，這個專案的核心可以濃縮成以下幾個點：

### `GameWorld.step(action)`

這是遊戲更新一幀的入口。只要理解這個方法，就理解了遊戲如何被 AI 使用。

### `FrameProcessor.preprocess(frame)`

這是原始畫面進入模型世界的入口。它決定模型看到的是什麼資料。

### `trim_episode_tail(...)`

這是資料品質控制的核心。它直接影響模型會不會被失敗尾段誤導。

### `train_model(...)`

這是把資料轉成模型能力的主流程。它負責真正完成學習。

### `_predict_action(...)`

這是模型輸出轉成遊戲動作的最後一步。沒有這一步，模型就只是靜態分類器，無法成為真正可執行的 AI 控制器。

---

## 這個系統的優點

第一，結構清楚。遊戲核心、資料錄製、前處理、模型與訓練流程各自分開，不會全部混在同一支程式裡。

第二，可展示性高。Flappy Bird 容易理解，加上 AI 自動遊玩與人機對戰，特別適合專題展示。

第三，資料透明。透過 inspect 與 session 機制，可以清楚知道模型是用什麼資料訓練出來的。

第四，可擴充性強。雖然目前是監督式 CNN，但未來可以延伸到更深模型、時間模型甚至強化學習。

第五，工程完整。這不只是模型定義，而是包含錄製、整理、訓練、推論與展示的完整系統。

---

## 目前的限制

第一，目前是模仿學習，不是強化學習，因此模型只能學到資料中出現過的行為。

第二，模型品質高度依賴錄製資料。如果資料量少、玩家操作不穩定、或資料分佈偏差，模型表現就會受影響。

第三，刪掉死亡尾段可以改善資料品質，但仍不能保證所有留下來的操作都是最佳決策。

第四，目前動作空間很簡單，只有 `WAIT` 與 `FLAP` 兩種。對 Flappy Bird 足夠，但本質上仍是簡化的分類問題。

第五，若要大量訓練或快速實驗，通常仍需要更好的 GPU 環境，例如 Colab。

---

## 未來可以怎麼擴充

如果想提高模型效果，可以增加更多玩家資料、做資料平衡分析，或換成更深的 CNN 架構。

如果想往研究方向延伸，可以把目前的監督式版本擴充成強化學習，例如 DQN 或 PPO。

如果想往展示與產品化發展，可以把錄製、訓練與檢視做成更完整的 GUI 或網頁介面。

如果想往教學用途發展，這個專案也非常適合拿來講解：

- 影像前處理
- CNN 基本原理
- 模仿學習
- 遊戲 AI
- 資料工程與模型訓練流程

---

## 總結

這個專案是一套以 Flappy Bird 為載體的 CNN 視覺學習系統。它的價值不在於重新做一個遊戲，而在於把一個遊戲完整轉換成可記錄、可訓練、可推論、可展示的 AI 平台。

它讓玩家能用自己的操作資料來訓練模型，透過畫面前處理與 CNN 分類器，讓 AI 從畫面中學會何時該拍翅。再加上資料清理、session 串接、模型儲存、訓練視覺化、AI 自動遊玩與人機對戰，整個專案形成了一個從資料到結果的完整工程流程。

就算完全不看程式碼，只要理解這份說明，也可以清楚知道它在做什麼：**讓電腦透過觀察遊戲畫面與玩家示範，學會控制 Flappy Bird。**
