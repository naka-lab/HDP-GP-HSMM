# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from HDPGPHSMM_segmentaion import GPSegmentation
import time
import matplotlib.pyplot as plt
import os
import glob

def learn( savedir, dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen ):
    gpsegm = GPSegmentation( dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen)

    files =  [ "Input_Data/testdata2dim_%03d.txt" % j for j in range(4) ]
    gpsegm.load_data( files )
    liks = []

    start = time.time()
    #iteration (default: 10)
    for it in range( 10 ):
        print( "-----", it, "-----" )
        gpsegm.learn()
        numclass = gpsegm.save_model( savedir )
        print( "lik =", gpsegm.calc_lik() )
        liks.append(gpsegm.calc_lik())
    print ("liks: ",liks)
    print( time.time()-start )

    #plot liks
    plt.clf()
    plt.plot( range(len(liks)), liks )
    plt.savefig( os.path.join( savedir,"liks.png") )

    return numclass


def recog( modeldir, savedir, dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen ):
    print ("class", initial_class)
    gpsegm = GPSegmentation( dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen)

    gpsegm.load_data( [ "Input_Data/testdata2dim_%03d.txt" % j for j in range(4) ] )
    gpsegm.load_model( modeldir )


    start = time.time()
    gpsegm.recog()
    print( "lik =", gpsegm.calc_lik() )
    print( time.time()-start )
    gpsegm.save_model( savedir )


def main():
    #parameters
    dim = 2
    gamma = 2.0
    eta = 5.0

    initial_class = 1

    avelen = 15
    maxlen = int(avelen + avelen*0.25)
    minlen = int(avelen*0.25)
    print(maxlen, minlen)
    skiplen = 1

    #learn
    print ( "=====", "learn", "=====" )
    recog_initial_class = learn( "learn/", dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen )
    #recognition
    print ( "=====", "recognition", "=====" )
    recog( "learn/", "recog/", dim, gamma, eta, recog_initial_class, avelen, maxlen, minlen, skiplen )
    return

if __name__=="__main__":
    main()
