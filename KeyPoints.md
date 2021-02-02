# [Day 03：撰寫第一支完整的 Keras 程式](https://ithelp.ithome.com.tw/articles/10233758)
機器學習的標準處理流程如下
![](https://ithelp.ithome.com.tw/upload/images/20200903/20001976UxH8Uf9GdY.png)

# ==[Day 04：神經網路的效能調校(Performance Tuning)](https://ithelp.ithome.com.tw/articles/10234059)==

# [Day 11：卷積神經網路(CNN) 剖析與視覺化](https://ithelp.ithome.com.tw/articles/10235547)
剖析
- 卷積(Convolution)使用的卷積核(kernel)並不是固定的，他們的矩陣值是經由訓練得到的最佳解。
- 抽象化(Abstraction)：透過卷積使影像抽象化，可逐漸把輪廓顯現出來，但不是每個卷積都有效。

# [Day 16：TensorFlow 2 Object Detection API 安裝](https://ithelp.ithome.com.tw/articles/10237443)
# ==[Day 21：Batch Normalization 筆記整理](https://ithelp.ithome.com.tw/articles/10241052)==
神經網路含很多(Deep)神經層時，常會在其中放置一些 Batch Normalization 層，顧名思義，它應該是作特徵縮放

Batch Normalization 就是作特徵縮放，將前一層的Output標準化，再轉至下一層，標準化公式如下：
$x_{new} = (x_{old} - \mu)/ \delta$

標準化的好處就是讓收斂速度快一點，不作的話，通常先導向梯度較大的方向前進，造成收斂路線曲折前進
![](https://ithelp.ithome.com.tw/upload/images/20200921/20001976Fu6rbIpyyg.png)

$\gamma$：控制規模縮放(Scale)
$\beta$：控制規模偏移(Shift)
![](https://ithelp.ithome.com.tw/upload/images/20200921/20001976PjALBzgnlW.jpg)

- 標準化訓練時逐批計算的。
- $\gamma、\beta$值是由訓練過程中計算出來的。

### Internal Covariate Shift
定義：假設我們要學習使用X預測Y時，當X的分配改變時，模型就逐漸失效了
例如：辨識狗的模型使用黃狗圖片作訓練資料集拿來便是花狗，效果就會不好。使用 由於使用 Batch Normalization(每批資料先被標準化)，就可以修正此問題

### Batch Normalization 優點

- 防止梯度消失(gradient vanishing)或梯度爆炸(gradient explosion)
- 收斂快(Train faster)。
- 可使用較大的學習率(Use higher learning rates)。
- 權重初始化較容易(Parameter initialization is easier)。
- Activation function 在訓練過程中易消失或提早停止學習，經過 Batch Normalization 會再復活(Makes activation functions viable by regulating the inputs to them)。
- 全面準確率提升(Better results overall)。
- 有類似 Dropout 的效果，防止過度擬合(It adds noise which reduces overfitting with a regularization effect)，所以，用了 Batch Normalization，就少一點 Dropout，避免效果過強，反而造成低度擬合(Underfitting)。

### 兩個資料集模擬【Internal Covariate Shift】現象
使用兩個資料集訓練一個網路
![](https://ithelp.ithome.com.tw/upload/images/20200921/200019769QR498Fsdu.png)

再使用兩個資料集訓練兩個網路，但共享權值
![](https://ithelp.ithome.com.tw/upload/images/20200921/20001976CotlXJRvRR.png)

結果(都有加入 Batch Normalization)
![](https://ithelp.ithome.com.tw/upload/images/20200921/20001976ocEfqOOZWy.png)

第三種模型：使用兩個資料集訓練兩個網路，但個別作 Batch Normalization，不共享權值。
![](https://ithelp.ithome.com.tw/upload/images/20200921/20001976HadnqE6LS0.png)

# [Day 22：Tensorflow Dataset 相關用法整理](https://ithelp.ithome.com.tw/articles/10241789)

# [Day 28：從直覺的角度初探強化學習](https://ithelp.ithome.com.tw/articles/10245605)

### 強化學習機制
![](https://ithelp.ithome.com.tw/upload/images/20200927/2000197639yFZJUioG.png)
- 代理人(Agent)：也就是遊戲中的玩家(Player)，他主要與環境互動，根據當時的狀態(State)以及之前得到的獎勵或懲罰，決定下一步的行動。
- 環境(Environment)：會根據代理人的行動(Action)，給予立即的獎勵或懲罰，統稱為獎勵(Reward)。
- 狀態(State)：也稱為觀察(Observation)，有時候代理人只能觀察到局部的狀態，例如，樸克牌遊戲21點(Black Jack)，莊家有一張牌是蓋住的，玩家是看不到。

### 簡單的強化學習架構
![](https://ithelp.ithome.com.tw/upload/images/20200928/20001976o1UTIJBIap.png)

採取物件導向設計(OOP)，總共有兩個類別，其職責(方法)如下：

- 環境(Environment)：類似遊戲本身。
  - Init (初始化)：需定義狀態空間(State Space)、獎勵(Reward)辦法、行動空間(Action Space)、狀態轉換(State Transition definition)。
  - Reset (重置)：回合(Episode)結束時，需重新開始，重置所有變數。
  - Step (步驟)：代理人行動後，會驅動下一步，環境會更新狀態，給予獎勵，並判斷回合是否結束及勝負。
  - Render (渲染)：更新畫面顯示。
- 代理人(Agent)：類似玩家。
  -  Act (行動)：代理人依據既定的策略以及面臨的狀態，採取行動，例如上、下、左、右。
  -  通常我們要訂定特定策略，就繼承基礎的代理人類別(base agent class)，在衍生的類別中，撰寫策略邏輯。

最後撰寫成一個類別或一段程式，稱之為【實驗】(Experiment)，用來建立環境、代理人兩個物件，讓系統動起來。

# [Day 30：取代資料科學家 -- AutoKeras 入門](https://ithelp.ithome.com.tw/articles/10246684)

AutoKeras 就是以 Keras 風格撰寫的 AutoML 套件，目前提供三類功能：

- 影像分類與迴歸(Image Classification and Regression)
- 文字分類與迴歸(Text Classification and Regression)
- 結構化資料分類與迴歸(Structured Data Classification and Regression)：即一般的表格資料，如 CSV、Excel、資料庫...等二維表格資料。

近期還會擴充 Time Series Forcasting, Object Detection, Image Segmentation 相關功能。