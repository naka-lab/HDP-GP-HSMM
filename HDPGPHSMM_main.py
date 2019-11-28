# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from HDPGPHSMM_segmentaion import GPSegmentation
import time
import matplotlib.pyplot as plt
import os
import glob

def learn( savedir, dim, gamma, eta, initial_class ):
    gpsegm = GPSegmentation( dim, gamma, eta, initial_class)

    files =  [ "Input_Data/testdata1dim_%03d.txt" % j for j in range(6) ]
    gpsegm.load_data( files )
    liks = []

    start = time.clock()
    #iteration (default: 10)
    for it in range( 10 ):
        print( "-----", it, "-----" )
        gpsegm.learn()
        numclass = gpsegm.save_model( savedir )
        print( "lik =", gpsegm.calc_lik() )
        liks.append(gpsegm.calc_lik())
    print ("liks: ",liks)
    print( time.clock()-start )

    #plot liks
    plt.clf()
    plt.plot( range(len(liks)), liks )
    plt.savefig( os.path.join( savedir,"liks.png") )

    return numclass


def recog( modeldir, savedir, dim, gamma, eta, initial_class ):
    print ("class", initial_class)
    gpsegm = GPSegmentation( dim, gamma, eta, initial_class)

    gpsegm.load_data( [ "Input_Data/testdata1dim_%03d.txt" % j for j in range(6) ] )
    gpsegm.load_model( modeldir )


    start = time.clock()
    for it in range(5):
        print( "-----", it, "-----" )
        gpsegm.recog()
        print( "lik =", gpsegm.calc_lik() )
    print( time.clock()-start )
    gpsegm.save_model( savedir )



def main():
    #parameters
    dim = 1
    gamma = 1.0
    eta = 10.0

    initial_class = 1
    #learn
    print ( "=====", "learn", "=====" )
    recog_initial_class = learn( "learn/", dim, gamma, eta, initial_class )
    #recognition
    print ( "=====", "recognition", "=====" )
    recog( "learn/", "recog/", dim, gamma, eta, recog_initial_class )
    return

if __name__=="__main__":
    main()
