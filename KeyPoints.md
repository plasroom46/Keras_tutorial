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