/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.fall2019.comp4980;


import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 *
 * @author John
 */
public class TestClass  {

    public static void main(String[] args) throws Exception {
        cnn();
    }

    public static void ae() throws Exception{
        AutoEncoder ae = new AutoEncoder(.001);
        Dataset ds = new Dataset(true);
        boolean training = true;
        while(training){
             ae.train(200, ds.trainingSet);
             training = false;
        }

        ae.test(ds);
    }

    public static void cnn() throws Exception{
        CNN cnn = new CNN(.001);
        Dataset cnnDataset = new Dataset(false);
        boolean training = true;
        if(training){
            cnn.train(100, cnnDataset.trainingSet);
        }

        cnn.test(cnnDataset);

    }
}
