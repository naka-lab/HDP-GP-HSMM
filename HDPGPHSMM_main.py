# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from HDPGPHSMM_segmentaion import GPSegmentation
import time
import matplotlib.pyplot as plt
import os

def learn( savedir ):
    gpsegm = GPSegmentation(2,1.0,10.0) #dim gamma eta

    files =  [ "testdata2d_%03d.txt" % j for j in range(4) ]
    gpsegm.load_data( files )
    liks = []
    
    start = time.clock()
    for it in range(10):
        print( "-----", it, "-----" )
        gpsegm.learn()
        gpsegm.save_model( savedir )
        print( "lik =", gpsegm.calc_lik() )
        liks.append(gpsegm.calc_lik())
    print ("liks: ",liks)
    print( time.clock()-start )
    
    #plot liks
    plt.clf()
    #plt.ylim(-18000, 1000)
    plt.plot( range(len(liks)), liks )
    #plt.show()
    plt.savefig( os.path.join( savedir,"lik.png") )
        
    return gpsegm.calc_lik()


def recog( modeldir, savedir ):
    gpsegm = GPSegmentation(2,1.0,10.0)

    gpsegm.load_data( [ "testdata2d_%03d.txt" % j for j in range(4) ] )
    gpsegm.load_model( modeldir )


    start = time.clock()
    for it in range(5):
        print( "-----", it, "-----" )
        gpsegm.recog()
        print( "lik =", gpsegm.calc_lik() )
    print( time.clock()-start )
    gpsegm.save_model( savedir )



def main():
    learn( "learn/" )
    #recog( "learn/" , "recog/" )
    return

if __name__=="__main__":
    main()