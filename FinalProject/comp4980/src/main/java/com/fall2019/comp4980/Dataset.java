package com.fall2019.comp4980;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Dataset {
     public static ArrayList<INDArray> trainingSet = new ArrayList<INDArray>();
     public static ArrayList<INDArray> testingSet = new ArrayList<INDArray>();
     public static Map<String, ArrayList<INDArray>> trainMap = new HashMap<String, ArrayList<INDArray>>();
     public static Map<String, ArrayList<INDArray>> testMap = new HashMap<String, ArrayList<INDArray>>();
     public static ArrayList<String> trainPath = new ArrayList<String>();
     public static ArrayList<String> testPath = new ArrayList<String>();


     private static File root;
     private static File rootDataset;
     private final static String allNames = "/people.csv";
     private final static String[] trainingPaths = {"0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"};
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
          setData(false);
          System.out.println("Testing set size: " + testingSet.size());
          System.out.println("Training set size: " + trainingSet.size());

          System.out.println("Dataset Initialized");
     }

     Dataset(boolean ae) throws Exception{
          System.out.println("Initializing Dataset...");
          System.out.println();
          setData(ae);
          System.out.println("Testing set size: " + testingSet.size());
          System.out.println("Training set size: " + trainingSet.size());

          System.out.println("Dataset Initialized");
     }
     public static void setData(boolean ae) throws Exception{

          String root = getDatasetRoot();
          String name;
          String image;
          String finalPath;
          File rootFile = new File(root);
          INDArray v_in;
          Img2INDArray.grayscale();


          for(File class_ : rootFile.listFiles()) {
               ArrayList<INDArray> listTest = new ArrayList<INDArray>();
               ArrayList<INDArray> listTrain = new ArrayList<INDArray>();
               String[] nameAndPath = class_.toString().split("/");
               name = nameAndPath[nameAndPath.length - 1];

               image = class_.toString() + "/" + name + "_";

               for (String postFix : trainingPaths) {

                    finalPath = image + postFix + ".jpg";
                    //System.out.println(finalPath);
                    trainPath.add(finalPath);

                    v_in = Img2INDArray.load_image(finalPath, width, height, offsetX, offsetY, 0, false);
                    if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                    listTrain.add(v_in);
                    trainingSet.add(v_in);

                    for(int i = 1; i < 4; i++) {

                         v_in = Img2INDArray.load_image(finalPath, width, height, i, i, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);

                         v_in = Img2INDArray.load_image(finalPath, width, height, -i, -i, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);

                         v_in = Img2INDArray.load_image(finalPath, width, height, i, -i, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);

                         v_in = Img2INDArray.load_image(finalPath, width, height, -i, i, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);

                         v_in = Img2INDArray.load_image(finalPath, width, height, -i, offsetY, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);

                         v_in = Img2INDArray.load_image(finalPath, width, height, i, offsetY, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);


                         v_in = Img2INDArray.load_image(finalPath, width, height, offsetX, i, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);

                         v_in = Img2INDArray.load_image(finalPath, width, height, offsetX, -i, 0, false);
                         if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                         listTrain.add(v_in);
                         trainingSet.add(v_in);
                    }

               }
               trainMap.put(name, listTrain);

               noise = 0.0;
               for(String postFix : testPaths){
                    finalPath = image + postFix + ".jpg";
                    testPath.add(finalPath);
                    v_in = Img2INDArray.load_image(finalPath, width, height, offsetX, offsetY, 0, false);
                    if(ae){v_in = v_in.ravel().reshape(1,v_in.length());}
                    listTest.add(v_in);
                    testingSet.add(v_in);
               }
               testMap.put(name, listTest);
          }
     }

     public static void setNoise(double n){
          noise = n;
     }
     private static String getDatasetRoot() throws Exception{
          String currentDir = new java.io.File( "." ).getCanonicalPath();
          String datasetRoot = currentDir + "/dataset/";
          return datasetRoot;
     }
}
