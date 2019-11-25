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
        ae();
    }

    public static void ae() throws Exception{

        Dataset ds = new Dataset(true);
        boolean trainNewModel = false;
        boolean testing = false;
        AutoEncoder ae;

        if(trainNewModel){
            ae = new AutoEncoder(.000033, true);
            ae.train(100, ds.trainingSet);
        }
        else{
            ae = new AutoEncoder(.000033, false);
        }
        if(testing){
            ae.test(ds);
        }
        else{
            ae.train(1000, ds.trainingSet);
            ae.test(ds);
        }
    }

    public static void cnn() throws Exception{
        Dataset cnnDataset = new Dataset(false);
    }
}
