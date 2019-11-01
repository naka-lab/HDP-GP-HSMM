# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:51:21 2018

@author: Nagano Masatoshi
"""
# python setup.py build_ext --inplace
# case in Python 2.6~
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import GaussianProcessMultiDim
import random
import math
import os
import sys
import time
#from scipy.misc import logsumexp
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from cymath import logsumexp

class GPSegmentation():
    # parameters
    MAX_LEN = 20
    MIN_LEN = 3
    AVE_LEN = 12
    SKIP_LEN = 1
    
    def __init__(self, dim, gamma, alpha):
        self.dim = dim
        self.numclass = 1
        self.segmlen = 3
        self.gps = [ GaussianProcessMultiDim.GPMD(dim) for i in range(self.numclass)]
        self.segm_in_class= [ [] for i in range(self.numclass)]
        self.segmclass = {}
        self.segments = []
        self.trans_prob = np.ones( (1,1) )
        self.trans_prob_bos = np.ones( 1 )
        self.trans_prob_eos = np.ones( 1 )
        self.all_numclass = []
        self.counter = 0
        self.SKIP_LEN = 1
        self.is_initialized = False
        
        # stick breaking process
        self.alpha = alpha
        self.beta = np.ones(1)
        self.gamma = gamma


    def load_data(self, z_s, classfile=None ):
        self.data = []
        self.segments = []
        self.is_initialized = False

        for z in z_s:
            y = np.loadtxt(z, dtype=np.float)
            segm = []
            self.data.append( y )

            i = 0
            while i<len(y):
                length = random.randint(self.MIN_LEN,self.MAX_LEN)

                if i+length+1>=len(y):
                    length = len(y)-i

                segm.append( y[i:i+length+1] )

                i+=length
            self.segments.append( segm )
            
            # ランダムに割り振る
            for i,s in enumerate(segm):
                c = random.randint(0,self.numclass-1)
                #c = 0
                self.segmclass[id(s) ] = c
                #self.segm_in_class[c].append( s )
        """
        # 各クラス毎に学習
        for c in range(self.numclass):
            self.update_gp( c )
        """
        # 遷移確率更新
        self.calc_trans_prob()

    #added based on GP-HSMM
    def load_model( self, basename ):
        # GP読み込み
        for c in range(self.numclass):
            filename = basename + "class%03d.npy" % c
            self.segm_in_class[c] = np.load( filename, allow_pickle=True)
            self.update_gp( c )

        # 遷移確率更新
        self.trans_prob = np.load( basename+"trans.npy", allow_pickle=True )
        self.trans_prob_bos = np.load( basename+"trans_bos.npy", allow_pickle=True )
        self.trans_prob_eos = np.load( basename+"trans_eos.npy", allow_pickle=True )


    def update_gp(self, c ):
        datay = []
        datax = []
        for s in self.segm_in_class[c]:
            datay += [ y for y in s ]
            datax += range(len(s))
        self.gps[c].learn( np.array(datax), datay )

    
    #ver_log
    def calc_emission_logprob( self, c, segm ):
        gp = self.gps[c]
        slen = len(segm)

        if len(segm) > 2:
            log_plen = (slen*math.log(self.AVE_LEN) + (-self.AVE_LEN)*math.log(math.e)) - (sum(np.log(np.arange(1,slen+1))))
            p = gp.calc_lik( np.arange(len(segm), dtype=np.float) , segm )
            return p + log_plen
        else:
            return 1.0e-100
    
    
    def save_model(self, basename ):
        if not os.path.exists(basename):
            os.mkdir( basename )

        for n,segm in enumerate(self.segments):
            classes = []
            cut_points = []
            for s in segm:
                c = self.segmclass[id(s)]
                classes += [ c for i in range(len(s)) ]
                cut_points += [0] * len(s)
                cut_points[-1] = 1
            np.savetxt( basename+"segm%03d.txt" % n, np.vstack([classes,cut_points]).T, fmt=str("%d") )
            
        for c in range(len(self.gps)):
            for d in range(self.dim):
                plt.clf()
                #plt.subplot( self.dim, self.numclass, c+d*self.numclass+1 )
                for data in self.segm_in_class[c]:
                    if self.dim==1:
                        plt.plot( range(len(data)), data, "o-" )
                    else:
                        plt.plot( range(len(data[:,d])), data[:,d], "o-" )
                    plt.ylim( -1, 1 )
                plt.savefig( basename+"class%03d_dim%03d.png" % (c, d) )
                    
        np.save( basename + "trans.npy" , self.trans_prob  )
        np.save( basename + "trans_bos.npy" , self.trans_prob_bos )
        np.save( basename + "trans_eos.npy" , self.trans_prob_eos )
        np.save( basename + "all_class.npy", self.segm_in_class[c])
        
        for c in range(self.numclass):
            np.save( basename+"class%03d.npy" % c, self.segm_in_class[c] )
            
            
    def forward_filtering(self, d ):
        T = len(d)
        a = np.zeros( (len(d), self.MAX_LEN, self.numclass) ) - 1.0e-100   # 前向き確率．対数で確率を保持．1.0e-100で確率0を近似的に表現．
        valid = np.zeros( (len(d), self.MAX_LEN, self.numclass) ) # 計算された有効な値かどうか．計算されていない場所の確率を0にするため．
        z = np.ones( T ) # 正規化定数
        
        for t in range(T):
            for k in range(self.MIN_LEN,self.MAX_LEN,self.SKIP_LEN):
                if t-k<0:
                    break

                segm = d[t-k:t+1]
                for c in range(self.numclass):
                    out_prob = self.calc_emission_logprob( c, segm )
                    foward_prob = 0.0
                    
                    # 遷移確率
                    tt = t-k-1
                    if tt>=0:
                        foward_prob = logsumexp( a[tt,:,:] + z[tt] + np.log(self.trans_prob[:,c]) ) + out_prob
                    else:
                        # 最初の単語
                        foward_prob = out_prob + math.log(self.trans_prob_bos[c])

                    if t==T-1:
                        # 最後の単語
                        foward_prob += math.log(self.trans_prob_eos[c])

                    # 正規化を元に戻す
                    a[t,k,c] = foward_prob
                    valid[t,k,c] = 1.0
                    if math.isnan(foward_prob):
                        print( "a[t=%d,k=%d,c=%d] became NAN!!" % (t,k,c) )
                        sys.exit(-1)
            # 正規化
            if t-self.MIN_LEN>=0:
                z[t] = logsumexp( a[t,:,:] )
                a[t,:,:] -= z[t]
                
        return np.exp(a)*valid
    

    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i


    def backward_sampling(self, a, d):
        T = a.shape[0]
        t = T-1
        segm = []
        segm_class = []

        c = -1
        while True:
            if t==T-1:
                transp = self.trans_prob_eos
            else:
                transp = self.trans_prob[:,c]

            idx = self.sample_idx( (a[t]*transp).reshape( self.MAX_LEN*self.numclass ))
            k = int(idx/self.numclass)
            c = idx % self.numclass
                     
            #バグ修正
            if t-k-1<=0:
                s = d[0:t+1]
            else:
                s = d[t-k:t+1]
            
            # パラメータ更新
            segm.insert( 0, s )
            segm_class.insert( 0, c )
            

            t = t-k-1
            self.counter += 1
            
            if t<=0:
                break
        
        return segm, segm_class


    def calc_trans_prob( self ):
        self.trans_prob = np.zeros( (self.numclass,self.numclass) )
        self.trans_prob_bos = np.zeros( self.numclass )
        self.trans_prob_eos = np.zeros( self.numclass )

        # 数え上げる
        for n,segm in enumerate(self.segments):
            if id(segm[0]) in self.segmclass:
                c_begin = self.segmclass[ id(segm[0]) ]
                self.trans_prob_bos[c_begin]+=1

            if id(segm[-1]) in self.segmclass:
                c_end = self.segmclass[ id(segm[-1]) ]
                self.trans_prob_eos[c_end]+=1

            for i in range(1,len(segm)):
                try:
                    cc = self.segmclass[ id(segm[i-1]) ]
                    c = self.segmclass[ id(segm[i]) ]
                except KeyError:
                    # gibss samplingで除かれているものは無視
                    continue
                self.trans_prob[cc,c] += 1

        # 事前確率
        self.trans_prob_bos += self.alpha * self.beta
        self.trans_prob_eos += self.alpha * self.beta

        for c in range(self.numclass):
            self.trans_prob[c,:] += self.alpha * self.beta

        # 正規化
        self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.numclass,1)
        self.trans_prob_bos = self.trans_prob_bos / np.sum( self.trans_prob_bos )
        self.trans_prob_eos = self.trans_prob_eos / np.sum( self.trans_prob_eos )


    def sample_num_states(self):

        # 補助変数uを計算
        u = []
        for n,segm in enumerate(self.segments):
            c = self.segmclass[ id(segm[0]) ]
            p = self.trans_prob_bos[c]
            u.append( random.random() * p )

            c = self.segmclass[ id(segm[-1]) ]
            p = self.trans_prob_eos[c]
            u.append( random.random() * p )

            for i in range(1,len(segm)):
                cc = self.segmclass[ id(segm[i-1]) ]
                c = self.segmclass[ id(segm[i]) ]
                p = self.trans_prob[cc,c]
                u.append( random.random() * p )

        # 不要な状態を削除
        beta = list( self.beta )
        for c in range(self.numclass)[::-1]: #後ろから一つずつ
            if len(self.segm_in_class[c])==0:
                self.numclass -= 1
                self.gps.pop()
                self.segm_in_class.pop()
                beta[-2] += beta[-1]
                beta.pop()
                #print ("pop!")
            else:
                break

        # 枝折りを繰り返す
        u_min = np.min( u )

        N = 0
        for c in range(self.numclass):
            N += len(self.segm_in_class[c])

        while self.alpha*beta[-1]/N > u_min:
            stick_len = beta[-1]
            rnd = np.random.beta(1,self.gamma)
            beta[-1] = stick_len * rnd
            beta.append( stick_len * (1-rnd) )
            self.numclass += 1
            self.gps.append( GaussianProcessMultiDim.GPMD(self.dim) )
            self.segm_in_class.append([])

        self.beta = np.array( beta )

        self.all_numclass.append(self.numclass)
        
        
    # list.remove( elem )だとValueErrorになる
    def remove_ndarray(self, lst, elem ):
        l = len(elem)
        for i,e in enumerate(lst):
            if len(e)!=l:
                continue
            if (e==elem).all():
                lst.pop(i)
                return
        raise ValueError( "ndarray is not found!!" )
        
        
    def learn(self):
        if self.is_initialized==False:
            # GPの学習
            for i in range(len(self.segments)):
                for s in self.segments[i]:
                    c = self.segmclass[id(s)]
                    self.segm_in_class[c].append( s )

            # 各クラス毎に学習
            for c in range(self.numclass):
                self.update_gp( c )

            self.is_initialized = True

        self.update(True)


    def update(self, learning_phase=True ):
        
        for i in range(len(self.segments)):
            print ("slice sampling")
            self.sample_num_states()
            d = self.data[i]
            segm = self.segments[i]

            for s in segm:
                c = self.segmclass[id(s)]
                self.segmclass.pop( id(s) )

                if learning_phase:
                    # パラメータ更新
                    self.remove_ndarray( self.segm_in_class[c], s )

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()

            start = time.clock()
            print ( "forward...", end="" )
            a = self.forward_filtering( d )

            print( "backward...", end="" )
            segm, segm_class = self.backward_sampling( a, d )
            print( time.clock()-start, "sec" )

            print( "Number of classified segments: [", end="")
            for s in self.segm_in_class:
                print( len(s), end=" " )
            print( "]" )
            
            self.segments[i] = segm

            for s,c in zip( segm, segm_class ):
                self.segmclass[id(s)] = c

                # パラメータ更新
                if learning_phase:
                    self.segm_in_class[c].append(s)

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()

        return


    #log
    def calc_lik(self):
        lik = 0
        for segm in self.segments:
            for s in segm:
                c = self.segmclass[id(s)]
                lik += self.gps[c].calc_lik( np.arange(len(s),dtype=np.float64) , s )
        return lik


    def get_num_class(self):
        n = 0
        for c in range(self.numclass):
            if len(self.segm_in_class[c])!=0:
                n += 1
        return n
    
    def recog(self):
        self.update(False)

