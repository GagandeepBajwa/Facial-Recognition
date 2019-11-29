package com.fall2019.comp4980;

import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.earlystopping.scorecalc.AutoencoderScoreCalculator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.sound.midi.SysexMessage;
import javax.xml.crypto.Data;
import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class AutoEncoder {

     static ComputationGraph model;
     static final int VECTOR_INPUT = 2500;
     static final int INPUT_NODES = 2500;
     static final int ENCODER_1_NODES = 1250;
     static final int ENCODER_2_NODES = 600;
     static final int ENCODER_3_NODES = 350;
     static final int ENCODER_4_NODES = 175;
     static final int EMBEDDED_NODES = 300;
     static final int DECODER_4_NODES = 175;
     static final int DECODER_3_NODES = 350;
     static final int DECODER_2_NODES = 600;
     static final int DECODER_1_NODES = 1250;
     static final int OUTPUT_NODES = 2500;
     private static ArrayList<Integer> accept = new ArrayList<Integer>();
     private static ArrayList<Integer> reject = new ArrayList<Integer>();
     public static Map<String, ArrayList<INDArray>> bioMap = new HashMap<>();
     private static Double threshold = 20.0;
     private static String aePath = "";

     AutoEncoder(double learningRate, boolean newModel)throws Exception{
          if(newModel){
               model = nn_init(learningRate);
          }
          else{
               model = ComputationGraph.load(new File("ae_60_1033.zip"), true);
               model.setLearningRate(learningRate);
          }

     }
     private static ComputationGraph nn_init(double learningRate){

          System.out.println("Initializing AutoEncoder...");
          // Some hyperparameters (variable) declarations here.


          ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                  .weightInit(WeightInit.XAVIER)
                  .activation(Activation.ELU)
                  .updater(new Adam(learningRate))
                  .l2(.00001)
                  .graphBuilder()

                  /* Begin. Start creating the neural network structure (Layers) here */

                  .addInputs("vector_in")             // Name of the layer(s) to receive the feature vector(s) for model input.

                  .addLayer("INPUT_I1", new DenseLayer.Builder()
                          .nIn(VECTOR_INPUT)
                          .nOut(INPUT_NODES)
                          .build(), "vector_in")

                  /* This is the ENCODER part */

                  .addLayer("ENCODER_H1", new DenseLayer.Builder()
                          .nIn(INPUT_NODES)
                          .nOut(ENCODER_1_NODES)
                          .build(), "INPUT_I1")


                  .addLayer("ENCODER_H2", new DenseLayer.Builder()
                          .nIn(ENCODER_1_NODES)
                          .nOut(ENCODER_2_NODES)
                          .build(), "ENCODER_H1")
/*
                  .addLayer("ENCODER_H3", new DenseLayer.Builder()
                          .nIn(ENCODER_2_NODES)
                          .nOut(ENCODER_3_NODES)
                          .build(), "ENCODER_H2")

                  .addLayer("ENCODER_H4", new DenseLayer.Builder()
                          .nIn(ENCODER_3_NODES)
                          .nOut(ENCODER_4_NODES)
                          .build(), "ENCODER_H3")
/*
                  /**    EMBEDDED LAYER    **/
                  .addLayer("EMBEDDED_01", new DenseLayer.Builder()
                          .nIn(ENCODER_2_NODES)
                          .nOut(EMBEDDED_NODES)
                          .build(), "ENCODER_H2")

//                  /** This is the DECODER part */
//                  .addLayer("DECODER_H4", new DenseLayer.Builder()
//                          .nIn(EMBEDDED_NODES)
//                          .nOut(DECODER_4_NODES)
//                          .build(), "EMBEDDED_01")
//
//                  .addLayer("DECODER_H3", new DenseLayer.Builder()
//                          .nIn(DECODER_4_NODES)
//                          .nOut(DECODER_3_NODES)
//                          .build(), "DECODER_H4")
//
                .addLayer("DECODER_H2", new DenseLayer.Builder()
                          .nIn(EMBEDDED_NODES)
                          .nOut(DECODER_2_NODES)
                          .build(), "EMBEDDED_01")

                  .addLayer("DECODER_H1", new DenseLayer.Builder()
                          .nIn(DECODER_2_NODES)
                          .nOut(DECODER_1_NODES)
                          .build(), "DECODER_H2")

                  .addLayer("OUTPUT_01", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)   // The loss is mean squared
                          .activation( Activation.IDENTITY )           // The output activation function for the output is IDENTITY
                          .nIn(DECODER_1_NODES)
                          .nOut(OUTPUT_NODES)
                          .build(), "DECODER_H1")

                  .setOutputs("OUTPUT_01")
                  .build();

          ComputationGraph net = new ComputationGraph(config);
          net.init();

          System.out.println("Model Initialized");

          return net;
     }

     public static void train(int epoch, ArrayList<INDArray> training_set) throws Exception {
          INDArray[] INPs = new INDArray[1];


          Double score = 0.0;
          Double batchScore = 0.0;
          System.out.println("Beginning training...");
          for( ; epoch>0; epoch--)
          {
               int count = 1;
               int batchCount = 1;
               score = 0.0;

               Collections.shuffle(training_set);
               for( INDArray t: training_set)
               {
                    batchCount++;

                    INPs[0] = t;
                    model.fit(INPs, INPs);
                    score += model.score();
                    batchScore += model.score();
                    if(count%30==0){

                         System.out.println( count + "/" + training_set.size() + "\t" + batchScore/30);
                         batchScore = 0.0;
                    }
                    count++;
               }

               score = score/training_set.size();
               System.out.println("EPOCH AVG : " +  score + "\t" + epoch + " to go!");
               if(epoch % 10   == 0){
                    aePath = "ae_" + epoch  + "_" + (int)(score*100000) +  ".zip";
                    model.save(new File(aePath));
               }
          }
          aePath = "ae_eot_" + (int)(score*100000)  + ".zip";
          model.save(new File(aePath));
     }

     public static void test(Dataset ds) throws Exception {
          Map<String, ArrayList<INDArray>> testMap = ds.testMap;
          INDArray[] INPs = new INDArray[1];
          INDArray unknownEmbeddedVector;
          ArrayList<String> correctAuthList = new ArrayList<>();
          int correctAuthentication = 0;
          int counter = 0;
          buildBioTemplates(ds);

          System.out.println("Beginning testing...");

          for(String person : testMap.keySet()) {
               System.out.println("Unknown person " + person);

               for (INDArray testImg : ds.testMap.get(person)) {
                    double trueNumerator = 1000;
                    double numerator = 1000;
                    Double denominator = 0.0;

                    INPs[0] = testImg;
                    Map<String, INDArray> activations = model.feedForward(INPs, false);
                    unknownEmbeddedVector = activations.get("EMBEDDED_01");

                    String closestTrue = "";
                    String closestOverall = "";

                    for (String personInBio : bioMap.keySet()) {
                         //System.out.println("Comparing against biometric " + personInBio);
                         double ret_dist = findClosestToPersonFromINDArray(personInBio, unknownEmbeddedVector);
                         if (ret_dist < numerator) {
                              closestOverall = personInBio;
                              numerator = ret_dist;
                         }
                         if(personInBio.compareTo(person) == 0 && ret_dist < trueNumerator){
                              closestTrue = personInBio;
                              trueNumerator = ret_dist;
                         }

                         denominator += ret_dist;
                    }
                    if (closestOverall.compareTo(person) == 0) {
                         correctAuthList.add(person);
                         correctAuthentication++;
                    }
                    double prob = 1.0 - numerator / denominator;
                    double trueProb = 1.0 - trueNumerator / denominator;
                    System.out.println("");
                    System.out.println("This is: " + person);
                    System.out.println("Most confident: " + closestOverall + " prob=" + prob);
                    System.out.println("Same person most confident: " + closestTrue + " prob=" + trueProb);
                    System.out.println("");
                    System.out.println("----------------------");
                    System.out.println("");
                    counter++;
               }
          }
          System.out.println("PERFORMANCE:\n" + (double)correctAuthentication/counter);
          System.out.println(correctAuthList);

     }

     private static Double findClosestToPersonFromINDArray(String person, INDArray unknownEV){
          // Finds the closest
          Double min = 1000.0;
          Double currentDistance;


          // For each of the bioTemplates for this person
          for(INDArray arr : bioMap.get(person)){
               currentDistance = Transforms.euclideanDistance(arr, unknownEV);

               if(currentDistance < min){
                    min = currentDistance;
               }
          }
          //System.out.println(min);
          return min;
     }

     private static Double calculateUncertainty(INDArray embeddedVector){
          Map<String, Double> minDistanceMap = new HashMap<>();
          Double distance = 0.0;
          for(String person : bioMap.keySet()){
               Double min = 1000.0;
               for(INDArray embedVector : bioMap.get(person)){
                    distance = Transforms.euclideanDistance(embedVector, embeddedVector);
                    System.out.println(distance);
               }
          }
          return distance;
     }

     private static void buildBioTemplates(Dataset ds){
          INDArray[] INPs = new INDArray[1];
          INDArray afterActivation;
          for(String person : ds.trainMap.keySet()){
               ArrayList<INDArray> listOfTemps = new ArrayList<>();
               for(INDArray arr : ds.trainMap.get(person)){
                    INPs[0] = arr;
                    Map<String, INDArray> activations = model.feedForward(INPs, false);
                    afterActivation = activations.get("EMBEDDED_01");
                    listOfTemps.add(afterActivation);
               }

               bioMap.put(person,listOfTemps);
          }
     }

     private static void calculateFARandFRR(){
          int sizeOfAccepts = accept.size();
          int sizeOfRejects = reject.size();
          Double totalFalseAccepts = 0.0;
          Double totalFalseRejects = 0.0;
          for(int val : accept){
               totalFalseAccepts+=val;
          }
          for(int val : reject){
               totalFalseRejects+=val;
          }
          System.out.println("FAR: " + totalFalseAccepts/sizeOfAccepts);
          System.out.println("FRR: " + totalFalseRejects/sizeOfRejects);
     }

     private static void authenticate(String name, boolean isTrue, Double val){
          // Is person a and is accepted True accept
          if(isTrue){
               if(val < threshold){
                    accept.add(0);
               }
               else{
                    // Rejected when it is the person, False Rejection
                    accept.add(1);
               }
          }
          // Is a forger
          else{
               if(val < threshold){
                    // is accepted anyway, False Acceptance
                    reject.add(1);
               }
               else{
                    reject.add(0);
               }
          }
     }

     private static Double getAverageDistance(ArrayList<Double> arr){
          Double total = 0.0;

          for(Double val : arr){
               total += val;
          }
          return total/arr.size();
     }

     private static Double getEucDistance(INDArray personExample){
          INDArray[] INPs = new INDArray[1];
          INDArray[] TEST = new INDArray[1];
          INPs[0] = personExample;
          Map<String,INDArray> activations = model.feedForward(INPs,false);
          TEST[0] = activations.get("EMBEDDED_01");
          return Transforms.euclideanDistance( INPs[0], TEST[0] );
     }
}
