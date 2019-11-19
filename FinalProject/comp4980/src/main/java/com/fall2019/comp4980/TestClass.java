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
        
        /**
         * Before loading an image. Call one of these methods to set the color type you want:
         * 
         * Img2INDArray.rgb() - default. Assumes a color image and returns an INDArray with 3 channels (RGB).
         * Img2INDArray.grayscale() - Converts the image to shades-of-gray. Returns an INDArray with 1 channel.
         * Img2INDArray.bw() - Converts an image to black-and-white. Returns an INDArray with 1 channel.
         * 
         */
        
        /**
         * Syntax:
         * 
         *  load_image(String filename, int width, int height, int offsetX, int offsetY, boolean preview)
         * 
         *      filename:   filename for image to load
         *      
         *      width:      the width you want the image to be. It will automatically be scaled to this width
         * 
         *      height:     the height you want the image to be. It will automatically be scaled to this height
         *      
         *      offsetX:    shift the image X pixels to the left or right
         *      
         *      offsetY:    shift the image Y pixels up or down
         * 
         *      noise:      a value between 0-1. In some cases it may be useful to add noise to the image.
         *                  A value of 0.0 will add no noise while a value of 1.0 will be 100% noisy image.
         * 
         *      preview:    if true, a pop-up of the scaled image will be displayed. Used for debugging
         * 
         *  returns:    returns a rank 4 INDArray with dimensions [[[[1,3,height,width]]]] that is already normalized. 
         *              The 3-dimension depth represents the RED, GREEN, BLUE channels of the image. 
         */    

        final String BASE_PATH = getDatasetRoot();


        double noise    = 0.4;  // Change to 0.50 for 50% noise in picture
        int width       = 250;
        int height      = 250;
        int offsetX     = 0;
        int offsetY     = 0;
        System.out.println(BASE_PATH);
            INDArray v_in;


            String class_ = "Frank_Marshall/";
            String image = "Frank_Marshall_0001.jpg";
            String im = BASE_PATH + class_ + image;
            /** Load image as a 3-channel RGB INDArray of shape { height x width x 3 } */
            Img2INDArray.rgb();
            v_in = Img2INDArray.load_image(im,width,height,offsetX,offsetY, noise, true);
            System.out.println( "RGB:\n" + v_in );
            System.out.println( v_in.shapeInfoToString() + "\n");

    }

    private static String getDatasetRoot() throws Exception{
        String currentDir = new java.io.File( "." ).getCanonicalPath();

        return currentDir + "/lfw-dataset/";
    }
}
