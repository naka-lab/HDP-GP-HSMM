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
import glob
#from scipy.misc import logsumexp
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from cymath import logsumexp, calc_forward_probability


class GPSegmentation():
    # parameters


    def __init__(self, dim, gamma, alpha, initial_class, avelen, maxlen, minlen, skiplen):
        self.dim = dim
        self.numclass = initial_class
        #self.segmlen = 3
        self.gps = [ GaussianProcessMultiDim.GPMD(dim) for i in range(self.numclass)]
        self.segm_in_class= [ [] for i in range(self.numclass)]
        self.segmclass = {}
        self.segments = []
        self.trans_prob = np.ones( (initial_class, initial_class) )
        self.trans_prob_bos = np.ones( initial_class )
        self.trans_prob_eos = np.ones( initial_class )
        self.all_numclass = []
        self.counter = 0
        self.is_initialized = False

        # stick breaking process
        self.alpha = alpha
        self.beta = np.ones(self.numclass)
        self.gamma = gamma

        self.MAX_LEN = maxlen
        self.MIN_LEN = minlen
        self.AVE_LEN = avelen
        self.SKIP_LEN = skiplen

        self.prior_table = [ i*math.log(self.AVE_LEN) -self.AVE_LEN - sum(np.log(np.arange(1,i+1))) for i in range(1,self.MAX_LEN+1) ]


    def load_data(self, z_s, classfile=None ):
        self.data = []
        self.segments = []
        self.is_initialized = False

        for n, z in enumerate(z_s):
            y = np.loadtxt(z, dtype=np.float)
            y = y.reshape( y.shape[0], -1 )
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
                self.segmclass[ (n,i) ] = c
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
            return math.log(1.0e-100)

    def calc_emission_logprob_all(self, d):
        T = len(d)
        # 出力確率を計算
        emission_prob_all = np.zeros( ( self.numclass, self.MAX_LEN, len(d)) )
        for c in range(self.numclass):
            params = self.gps[c].predict( range(self.MAX_LEN) )
            for k in range(self.MAX_LEN):
                for dim in range( self.dim ):
                    mu = params[dim][0][k]
                    sig = params[dim][1][k]
                    emission_prob_all[c, k, 0:T-k] += -math.log(math.sqrt( 2*math.pi*sig**2)) - (d[k:,dim]-mu)**2 / (2*sig**2)

        # 累積確率にする
        for k in range(1, self.MAX_LEN):
            emission_prob_all[:, k, :] += emission_prob_all[:, k-1, :]

        for k in range(self.MAX_LEN):
            emission_prob_all[:,k,:] += self.prior_table[k]

        return emission_prob_all

    def save_model(self, basename ):
        if not os.path.exists(basename):
            os.mkdir( basename )

        for n,segm in enumerate(self.segments):
            classes = []
            cut_points = []
            for i, s in enumerate(segm):
                c = self.segmclass[(n,i)]
                classes += [ c for j in range(len(s)) ]
                cut_points += [0] * len(s)
                cut_points[-1] = 1
            np.savetxt( basename+"segm%03d.txt" % n, np.vstack([classes,cut_points]).T, fmt=str("%d") )
            #np.savetxt( basename+"beta%03d.txt" % n, np.array(self.beta), fmt=str("%d") )
            np.savetxt( basename+"beta%03d.txt" % n, np.array(self.beta) )

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
                plt.close()

        np.save( basename + "trans.npy" , self.trans_prob  )
        np.save( basename + "trans_bos.npy" , self.trans_prob_bos )
        np.save( basename + "trans_eos.npy" , self.trans_prob_eos )
        np.save( basename + "all_class.npy", self.segm_in_class[c])

        for c in range(self.numclass):
            #np.save( basename+"class%03d.npy" % c, self.segm_in_class[c] )
            np.save( basename+"class%03d.npy" % c, np.array(self.segm_in_class[c], dtype=object) )

        return self.numclass


    def calc_vitervi_path(self, d ):
        T = len(d)
        log_a = np.zeros( (len(d), self.MAX_LEN, self.numclass) ) - 1000000   # 前向き確率．対数で確率を保持．1.0e-100で確率0を近似的に表現．
        valid = np.zeros( (len(d), self.MAX_LEN, self.numclass) ) # 計算された有効な値可どうか．計算されていない場所の確率を0にするため．
        z = np.ones( T ) # 正規化定数
        path_kc = -np.ones(  (len(d), self.MAX_LEN, self.numclass, 2), dtype=np.int32 )
        emission_prob_all = self.calc_emission_logprob_all( d )

        # 前向き確率計算
        for t in range(T):
            for k in range(self.MIN_LEN,self.MAX_LEN,self.SKIP_LEN):
                if t-k<0:
                    break

                #segm = d[t-k:t+1]
                for c in range(self.numclass):
                    out_prob = emission_prob_all[c,k,t-k]
                    #out_prob = self.calc_emission_logprob( c, segm )
                    foward_prob = 0.0

                    # 遷移確率
                    tt = t-k-1
                    if tt>=0:
                        prev_prob = log_a[tt,:,:] + z[tt] + np.log(self.trans_prob[:,c])

                        # 最大値を取る
                        idx = np.argmax( prev_prob.reshape( self.MAX_LEN*self.numclass ))
                        kk = int(idx/self.numclass)
                        cc = idx % self.numclass

                        path_kc[t, k, c, 0] = kk
                        path_kc[t, k, c, 1] = cc

                        foward_prob = prev_prob[kk, cc] + out_prob
                    else:
                        # 最初の単語
                        foward_prob = out_prob + math.log(self.trans_prob_bos[c])

                        path_kc[t, k, c, 0] = t+1
                        path_kc[t, k, c, 1] = -1


                    if t==T-1:
                        # 最後の単語
                        foward_prob += math.log(self.trans_prob_eos[c])


                    log_a[t,k,c] = foward_prob
                    valid[t,k,c] = 1.0
                    if math.isnan(foward_prob):
                        print( "a[t=%d,k=%d,c=%d] became NAN!!" % (t,k,c) )
                        sys.exit(-1)
            # 正規化
            if t-self.MIN_LEN>=0:
                z[t] = logsumexp( log_a[t,:,:] )
                log_a[t,:,:] -= z[t]
                #z[t] = logsumexp( a[t,:,:] )
                #a[t,:,:] -= z[t]

        # バックトラック
        t = T-1
        idx = np.argmax( log_a[t].reshape( self.MAX_LEN*self.numclass ))
        k = int(idx/self.numclass)
        c = idx % self.numclass

        segm = [ d[t-k:t+1] ]
        segm_class = [ c ]

        while True:
            kk, cc = path_kc[t, k, c]

            t = t-k-1
            k = kk
            c = cc

            if t<=0:
                break

            if t-k-1<=0:
                #先頭
                s = d[0:t+1]
            else:
                #先頭以外
                s = d[t-k:t+1]

            segm.insert( 0, s )
            segm_class.insert( 0, c )

        return segm, segm_class


    def forward_filtering(self, d ):
        emission_prob_all = self.calc_emission_logprob_all( d )
        forward_prob = calc_forward_probability( emission_prob_all, self.trans_prob, self.trans_prob_bos, self.trans_prob_eos, len(d), self.MIN_LEN, self.SKIP_LEN, self.MAX_LEN, self.numclass )
        return forward_prob

        """
        T = len(d)
        log_a = np.log(np.zeros( (len(d), self.MAX_LEN, self.numclass) ) + 1.0e-100 )  # 前向き確率．対数で確率を保持．1.0e-100で確率0を近似的に表現．
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
                        foward_prob = logsumexp( log_a[tt,:,:] + z[tt] + np.log(self.trans_prob[:,c]) ) + out_prob
                    else:
                        # 最初の単語
                        foward_prob = out_prob + math.log(self.trans_prob_bos[c])

                    if t==T-1:
                        # 最後の単語
                        foward_prob += math.log(self.trans_prob_eos[c])

                    # 正規化を元に戻す
                    log_a[t,k,c] = foward_prob
                    valid[t,k,c] = 1.0
                    if math.isnan(foward_prob):
                        print( "a[t=%d,k=%d,c=%d] became NAN!!" % (t,k,c) )
                        sys.exit(-1)
            # 正規化
            if t-self.MIN_LEN>=0:
                z[t] = logsumexp( log_a[t,:,:] )
                log_a[t,:,:] -= z[t]

        return np.exp(log_a)*valid
        """


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
            if (n,0) in self.segmclass:
                c_begin = self.segmclass[ (n,0) ]
                self.trans_prob_bos[c_begin]+=1

            if (n,len(segm)-1) in self.segmclass:
                c_end = self.segmclass[ (n,len(segm)-1) ]
                self.trans_prob_eos[c_end]+=1

            for i in range(1,len(segm)):
                try:
                    cc = self.segmclass[ (n,i-1) ]
                    c = self.segmclass[ (n,i) ]
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
            c = self.segmclass[ (n, 0) ]
            p = self.trans_prob_bos[c]
            u.append( random.random() * p )

            c = self.segmclass[ (n, len(segm)-1) ]
            p = self.trans_prob_eos[c]
            u.append( random.random() * p )

            for i in range(1,len(segm)):
                cc = self.segmclass[ (n, i-1) ]
                c = self.segmclass[ (n, i) ]
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
            for n in range(len(self.segments)):
                for i, s in enumerate(self.segments[n]):
                    c = self.segmclass[(n, i)]
                    self.segm_in_class[c].append( s )

            # 各クラス毎に学習
            for c in range(self.numclass):
                self.update_gp( c )

            self.is_initialized = True

        self.update(True)


    def update(self, learning_phase=True ):

        for n in range(len(self.segments)):
            if learning_phase:
                self.sample_num_states()

            d = self.data[n]
            segm = self.segments[n]

            for i, s in enumerate(segm):
                c = self.segmclass[(n, i)]
                self.segmclass.pop( (n, i) )

                if learning_phase:
                    # パラメータ更新
                    self.remove_ndarray( self.segm_in_class[c], s )

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()

            start = time.time()
            print ( "forward...", end="" )
            a = self.forward_filtering( d )

            print( "backward...", end="" )
            segm, segm_class = self.backward_sampling( a, d )
            print( time.time()-start, "sec" )

            print( "Number of classified segments: [", end="")
            for s in self.segm_in_class:
                print( len(s), end=" " )
            print( "]" )

            self.segments[n] = segm

            start = time.time()
            for i, (s,c) in enumerate(zip( segm, segm_class )):
                self.segmclass[(n, i)] = c

                # パラメータ更新
                if learning_phase:
                    self.segm_in_class[c].append(s)

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()

            print( "parameter update...", time.time()-start, "sec" )

        """
        If you update the hyperparameter, uncomment the next two lines.
        """
        #print("<<< update the hyperparameter >>>")
        #for c in range(self.numclass):
        #    self.gps[c].estimate_hyperparams(100)

        return


    #log
    def calc_lik(self):
        lik = 0
        for n, segm in enumerate(self.segments):
            for i, s in enumerate(segm):
                c = self.segmclass[(n, i)]
                lik += self.gps[c].calc_lik( np.arange(len(s),dtype=float) , s )
        return lik


    def get_num_class(self):
        n = 0
        for c in range(self.numclass):
            if len(self.segm_in_class[c])!=0:
                n += 1
        return n

    def recog(self):
        #self.update(False)
        self.segmclass.clear()

        for n, d in enumerate(self.data):
            # viterviで解く
            segms, classes = self.calc_vitervi_path( d )
            self.segments[n] = segms
            for i, (s, c) in enumerate(zip( segms, classes )):
                self.segmclass[(n, i)] = c
