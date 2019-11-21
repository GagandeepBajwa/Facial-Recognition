/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.fall2019.comp4980;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author John
 */


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
public class Img2INDArray {
    
    private static int image_type = BufferedImage.TYPE_INT_RGB;
    
    private static int channels = 3;
    
    private static double normalizer = 255.0;
    
    
    public static void grayscale()
    {
        image_type = BufferedImage.TYPE_BYTE_GRAY;
        channels = 1;
        normalizer = 255.0;
    }
    
    
    public static void bw()
    {
        image_type = BufferedImage.TYPE_BYTE_BINARY;
        channels = 1;
        normalizer = 1.0;
    }
    
    
    public static void rgb()
    {
        image_type = BufferedImage.TYPE_INT_RGB;
        channels = 3;
        normalizer = 255.0;
    }
    
    public static INDArray load_image(String filename,int width, int height, int offsetX, int offsetY, double level, boolean preview) throws Exception
    {

           BufferedImage img = ImageIO.read( new File( filename ) );
                    
           return preProcess( width, height, img, offsetX, offsetY, level, preview );
    }
    
    public static INDArray preProcess( int width, int height, BufferedImage img, int offsetX, int offsetY, double level, boolean preview )
    {
        corrupt(img,level);
        
        BufferedImage bi = new BufferedImage(width, height, image_type);
              
        Graphics bg = bi.getGraphics();
        bg.drawImage(img, offsetX, offsetY, width,height, null);   
        bg.dispose();
        
        if ( preview )
            new PreviewImage(bi).setVisible(true);
        
        return convertImgToINDArray( bi );
    }
    
    public static void corrupt(BufferedImage img, double level)
    {
        for(int x=0; x<img.getWidth(); x++)
            for(int y=0; y<img.getHeight(); y++)
                if ( Math.random()<level ) img.setRGB( x,y, (int)(16777216.0*Math.random()) ); 
    }
    
    public static INDArray convertImgToINDArray( BufferedImage img )
    {
        INDArray v = Nd4j.zeros( 1, channels, img.getHeight(), img.getWidth() );
        
        getImageBytes( v, img );
              
        return v;
    }
    
    public static void getImageBytes(INDArray v, BufferedImage bi)
    {
        int[] pixel;

        for (int y = 0; y < bi.getWidth(); y++) 
        {
            for (int x = 0; x < bi.getHeight(); x++) 
            {
                pixel = bi.getRaster().getPixel(y, x, new int[channels]);
                
                for(int c=0; c<channels; c++)
                    v.putScalar(0, c, x, y, Double.valueOf(pixel[c])/normalizer );
            }
        }
        
    }    
    
}
