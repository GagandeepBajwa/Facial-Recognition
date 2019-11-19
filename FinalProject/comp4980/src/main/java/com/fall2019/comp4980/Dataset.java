package com.fall2019.comp4980;

public class Dataset {
     public static String root;

     Dataset(){
          System.out.println("Initializing Dataset...");
          try {
               root = getDatasetRoot();
          }
          catch(Exception e){
               System.out.println(e.toString());
          }
          System.out.println("Dataset Initialized");
     }
     private static String getDatasetRoot() throws Exception{
          String currentDir = new java.io.File( "." ).getCanonicalPath();

          return currentDir + "/lfw-dataset/";
     }
}
