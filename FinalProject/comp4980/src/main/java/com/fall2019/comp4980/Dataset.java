package com.fall2019.comp4980;

import java.io.File;

public class Dataset {
     private static File root;
     private static File rootDataset;

     Dataset(){
          System.out.println("Initializing Dataset...");
          try {
               root = getDatasetRoot();

               rootDataset = new File(root.toString() + "/lfw-deepfunneled");

               for(File file : rootDataset.listFiles()){
                    System.out.println(file.toString());
               }
          }
          catch(Exception e){
               System.out.println(e);
               System.exit(0);
          }
          System.out.println("Dataset Initialized");
     }
     private static File getDatasetRoot() throws Exception{
          String currentDir = new java.io.File( "." ).getCanonicalPath();
          File datasetRoot = new File( currentDir + "/lfw-dataset" );
          return datasetRoot;
     }
}
