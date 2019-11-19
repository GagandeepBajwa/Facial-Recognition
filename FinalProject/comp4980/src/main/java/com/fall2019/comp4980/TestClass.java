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

    public static void main(String[] args) throws Exception
    {
        final String allNames = "/lfw_allnames.csv";
        final String mismatchPairsDevTest = "/mismatchPairsDevTest.csv";
        final String mismatchpairsDevTrain = "/matchpairsDevTrain.csv";
        final String matchpairsDevTest = "/matchpairsDevTest.csv";
        final String matchpairsDevTrain = "/matchpairsDevTrain.csv";
        final String pairs = "/pairs.csv";
        final String people = "/people.csv";
        final String peopleDevTest = "/peopleDevTest.csv";
        final String peopleDevTrain = "/peopleDevTrain.csv";

        double noise    = 0.4;  // Change to 0.50 for 50% noise in picture
        int width       = 250;
        int height      = 250;
        int offsetX     = 0;
        int offsetY     = 0;

        INDArray v_in;

        /** Load image as a 3-channel RGB INDArray of shape { height x width x 3 } */
//        Img2INDArray.rgb();
//        v_in = Img2INDArray.load_image(im,width,height,offsetX,offsetY, noise, false);
//        System.out.println( "RGB:\n" + v_in );
//        System.out.println( v_in.shapeInfoToString() + "\n");

        // AutoEncoder ae = new AutoEncoder(.001);
        Dataset ts = new Dataset();
        ts.getCSV(allNames);
        ts.getCSV(mismatchPairsDevTest);
        ts.getCSV(mismatchpairsDevTrain);
        ts.getCSV(matchpairsDevTest);
        ts.getCSV(matchpairsDevTrain);
        ts.getCSV(pairs);
        ts.getCSV(people);
        ts.getCSV(peopleDevTest);
        ts.getCSV(peopleDevTrain);



        //ae.train(10000, )


    }


}
