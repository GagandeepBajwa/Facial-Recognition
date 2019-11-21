package com.fall2019.comp4980;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.util.ArrayList;

public class Dataset {
     public static ArrayList<INDArray> trainingSet = new ArrayList<INDArray>();
     public static ArrayList<INDArray> testingSet = new ArrayList<INDArray>();;
     private static File root;
     private static File rootDataset;
     private final static String allNames = "/people.csv";
     private final static String[] testPaths = {"0011","0012","0013","0014","0015"};
     static int width = 50;
     static int height = 50;
     static int offsetX = 0;
     static int offsetY = 0;
     static double noise = 0.2;

     /* Will set a dataset for cnn by default. If boolean provided, will ravel
     and reshape both the training and testing data.
      */
     Dataset() throws Exception{
          System.out.println("Initializing Dataset...");
          System.out.println();
          System.out.println("Testing set size: " + testingSet.size());
          System.out.println("Training set size: " + trainingSet.size());
          setData(false);
          System.out.println("Dataset Initialized");
     }

     Dataset(boolean ae) throws Exception{
          System.out.println("Initializing Dataset...");
          System.out.println();
          System.out.println("Testing set size: " + testingSet.size());
          System.out.println("Training set size: " + trainingSet.size());

          System.out.println("Dataset Initialized");
     }
     public static void setData(boolean ae) throws Exception{
          String[] trainingPaths = {"0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"};
          String root = getDatasetRoot();
          String name;
          String image;
          String finalPath;
          File rootFile = new File(root);
          INDArray v_in;
          Img2INDArray.rgb();

          for(File class_ : rootFile.listFiles()) {
               String[] nameAndPath = class_.toString().split("/");
               name = nameAndPath[nameAndPath.length - 1];
               image = class_.toString() + "/" + name + "_";

               for (String postFix : trainingPaths) {
                    finalPath = image + postFix + ".jpg";
                    v_in = Img2INDArray.load_image(finalPath, width, height, offsetX, offsetY, noise, false);
                    if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                    trainingSet.add(v_in);
               }

               for(String postFix : testPaths){
                    finalPath = image + postFix + ".jpg";
                    v_in = Img2INDArray.load_image(finalPath, width, height, offsetX, offsetY, noise, false);
                    if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                    testingSet.add(v_in);
               }
          }
     }
     private static String getDatasetRoot() throws Exception{
          String currentDir = new java.io.File( "." ).getCanonicalPath();
          String datasetRoot = currentDir + "/dataset/";
          return datasetRoot;
     }


}
