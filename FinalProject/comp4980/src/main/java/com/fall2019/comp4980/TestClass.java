/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.fall2019.comp4980;


/**
 *
 * @author John
 */
public class TestClass  {

    public static void main(String[] args) throws Exception {
        cnn();
        //ae();
    }

    public static void ae() throws Exception{

        Dataset ds = new Dataset(true);
        boolean trainNewModel = false;
        boolean testing = true;

        AutoEncoder ae;

        if(trainNewModel){
            ae = new AutoEncoder(.001, true);
            ae.train(200, ds.trainingSet);
        }
        else{
            ae = new AutoEncoder(.000333, false);
        }
        if(testing){
            ae.test(ds);
        }
        else{
            ae.train(100, ds.trainingSet);
            ae.test(ds);
        }
    }

    public static void cnn() throws Exception{
        CNN cnn = new CNN(.000333, false);
        Dataset cnnDataset = new Dataset(false);
        boolean training = false;
        if(training){
            cnn.train2(10000, cnnDataset.trainingSet, cnnDataset);
        }

        cnn.test(cnnDataset);

    }
}
