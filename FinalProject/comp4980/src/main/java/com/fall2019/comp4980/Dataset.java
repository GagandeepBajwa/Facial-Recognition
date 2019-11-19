package com.fall2019.comp4980;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.util.ArrayList;

public class Dataset {
     private static File root;
     private static File rootDataset;
     INDArray[] dataset;
     Dataset(){
          System.out.println("Initializing Dataset...");
          try {
               root = getDatasetRoot();
               rootDataset = new File(root.toString() + "/lfw-deepfunneled");
               dataset = initDataset(rootDataset);

          }
          catch(Exception e){
               System.out.println(e);
               System.exit(0);
          }
          System.out.println("Dataset Initialized");
     }

     private static INDArray[] initDataset(File rootDataset){
          long numberOfClasses = rootDataset.length();
          System.out.println(numberOfClasses);
          for(File file : rootDataset.listFiles()){
          }
          return null;
     }

     public static ArrayList<String> getCSV(String fileToRead){
          ArrayList<String> container = new ArrayList<String>();
          try {
               File root = getDatasetRoot();
               String mismatch = root.toString() + fileToRead;
               String line;
               BufferedReader mismatchFile = new BufferedReader(new FileReader(mismatch));
               boolean firstline = true;

               while((line = mismatchFile.readLine()) != null){
                    if(firstline) { firstline = false; continue; }
                    container.add(line);
               }
               System.out.println(fileToRead);
               for(String l : container){
                    System.out.println(l);
               }
               return null;
          }catch (Exception e){
               e.printStackTrace();
               System.exit(0);
          }
          return container;
     }

     private static File getDatasetRoot() throws Exception{
          String currentDir = new java.io.File( "." ).getCanonicalPath();
          File datasetRoot = new File( currentDir + "/lfw-dataset" );
          return datasetRoot;
     }
}
