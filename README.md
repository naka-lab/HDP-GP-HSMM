# HDP-GP-HSMM

階層ディリクレ過程とガウス過程，隠れセミマルコフモデルを用いた時系列データの分節化の実装です．
ガウス過程の計算は，Cythonと計算のキャッシュを利用して高速化しています．
詳細は以下の論文を参照してください．

Masatoshi Nagano, Tomoaki Nakamura, Takayuki Nagai, Daichi Mochihashi, Ichiro Kobayashi and Masahide Kaneko, “Sequence Pattern Extraction by Segmenting Time Series Data Using GP-HSMM with Hierarchical Dirichlet Process”, 
2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 
pp. 4067-4074, Oct. 2018 [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594029)

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
