# HDP-GP-HSMM

階層ディリクレ過程とガウス過程，隠れセミマルコフモデルを用いた時系列データの分節化の実装です．
ガウス過程の計算は，Cythonと計算のキャッシュを利用して高速化しています．
詳細は以下の論文を参照してください．

Masatoshi Nagano, Tomoaki Nakamura, Takayuki Nagai, Daichi Mochihashi, Ichiro Kobayashi and Masahide Kaneko, “Sequence Pattern Extraction by Segmenting Time Series Data Using GP-HSMM with Hierarchical Dirichlet Process”,
2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),
pp. 4067-4074, Oct. 2018 [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594029)

さらに以下の文献で提案された高速化法を導入，計算のCython化，逆行列演算の工夫により，従来のGP-HSMMに比べ高速な計算が可能です．

川村 美帆，佐々木 雄一，中村 裕一，"GP-HSMM の尤度計算並列化による高速な身体動作の分節化方式"，計測自動制御学会 システムインテグレーション部門講演会，1A4-08，2021


## 実行方法

```
python HDPGPHSMM_main.py
```

Cythonで書かれたプログラムは実行時に自動的にコンパイルされます．
WindowsのVisual Studioのコンパイラでエラーが出る場合は，

```
(Pythonのインストールディレクトリ)/Lib/distutils/msvc9compiler.py
```

の`get_build_version()`内の

```
majorVersion = int(s[:-2]) - 6
```

を使いたいVisual Studioのバージョンに書き換えてください．
VS2012の場合は，`majorVersion = 11`となります．

### 使用上の注意

使用する際は，以下のハイパーパラメータに注意してください．
Input Dataを変更した場合は, `*` の付いているパラメータは要変更．
#### hypara1
```
GaussianProcess.pyの
covariance_func 関数内の

theta0 ~ theta3
によってカーネルを決定する．

<default>
theta0 = 1.0
theta1 = 1.0
theta2 = 0.0
theta3 = 16.0
```
#### hypara2
```
HDPGPHSMM_segmentation.pyの
GPSegmentation クラスの

* MAX_LEN: 最大の分節長 K (default: 20)
* MIN_LEN: 最小の分節長   (default: 3)
* AVE_LEN: 分節長を決めるポアソン分布のパラメータ (default: 12)
SKIP_LEN: forward filtering-backward samplingで計算するtの間隔 (default: 1)
```

#### hypara3
```
HDPGPHSMM_main.pyの
main 関数の

* dim: Input Dataの次元 (default: 2)
gamma: Stick Breaking ProcessのBeta分布のparameter (default: 1.0)
eta: Hierarchical Dirichlet processのparameter (default: 10.0)
```


# LICENSE
This program is freely available for free non-commercial use.
If you publish results obtained using this program, please cite:

```
@INPROCEEDINGS{8594029,
author={M. {Nagano} and T. {Nakamura} and T. {Nagai} and D. {Mochihashi} and I. {Kobayashi} and M. {Kaneko}},
booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
title={Sequence Pattern Extraction by Segmenting Time Series Data Using GP-HSMM with Hierarchical Dirichlet Process},
year={2018},
volume={},
number={},
pages={4067-4074},
keywords={Bayes methods;feature extraction;Gaussian processes;hidden Markov models;image motion analysis;image sampling;image segmentation;learning (artificial intelligence);nonparametric statistics;time series;continuous time-series data;semiMarkov model;Gaussian processes;nonparametric models;unit motion patterns;complicated continuous motion;nonparametric Bayesian model;hierarchical Dirichlet process;hierarchical Dirichlet processes-Gaussian process;HDP-GP-HSMM;motion-capture data;sequence pattern extraction;time series data;continuous information;unit motions;unsupervised segmentation;Hidden Markov models;Motion segmentation;Gaussian processes;Bayes methods;Data models;Trajectory;Kernel},
doi={10.1109/IROS.2018.8594029},
ISSN={2153-0866},
month={Oct},}
```
