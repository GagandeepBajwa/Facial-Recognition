package com.fall2019.comp4980;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

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

     private static File getDatasetRoot() throws Exception{
          String currentDir = new java.io.File( "." ).getCanonicalPath();
          File datasetRoot = new File( currentDir + "/lfw-dataset" );
          return datasetRoot;
     }
}
