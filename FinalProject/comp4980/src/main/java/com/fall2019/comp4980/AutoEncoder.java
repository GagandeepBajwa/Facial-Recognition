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

import java.io.File;
import java.util.ArrayList;
import java.util.Map;

public class AutoEncoder {

     static ComputationGraph model;
     static final int VECTOR_INPUT = 7500;
     static final int INPUT_NODES = 7500;
     static final int ENCODER_1_NODES = 1000;
     static final int ENCODER_2_NODES = 100;
     static final int ENCODER_3_NODES = 10;
     static final int ENCODER_4_NODES = 5;
     static final int EMBEDDED_NODES = 2;
     static final int DECODER_4_NODES = 5;
     static final int DECODER_3_NODES = 10;
     static final int DECODER_2_NODES = 100;
     static final int DECODER_1_NODES = 1000;
     static final int OUTPUT_NODES = 7500;

     AutoEncoder(double learningRate){
          model = nn_init(learningRate);
     }
     private static ComputationGraph nn_init(double learningRate){

          System.out.println("Initializing AutoEncoder...");
          // Some hyperparameters (variable) declarations here.


          ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                  .weightInit(WeightInit.XAVIER)
                  .activation(Activation.ELU)
                  .updater(new Adam(learningRate))

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

                  .addLayer("ENCODER_H3", new DenseLayer.Builder()
                          .nIn(ENCODER_2_NODES)
                          .nOut(ENCODER_3_NODES)
                          .build(), "ENCODER_H2")

                  .addLayer("ENCODER_H4", new DenseLayer.Builder()
                          .nIn(ENCODER_3_NODES)
                          .nOut(ENCODER_4_NODES)
                          .build(), "ENCODER_H3")

                  /**    EMBEDDED LAYER    **/
                  .addLayer("EMBEDDED_01", new DenseLayer.Builder()
                          .nIn(ENCODER_4_NODES)
                          .nOut(EMBEDDED_NODES)
                          .build(), "ENCODER_H4")

                  /** This is the DECODER part */
                  .addLayer("DECODER_H4", new DenseLayer.Builder()
                          .nIn(EMBEDDED_NODES)
                          .nOut(DECODER_4_NODES)
                          .build(), "EMBEDDED_01")

                  .addLayer("DECODER_H3", new DenseLayer.Builder()
                          .nIn(DECODER_4_NODES)
                          .nOut(DECODER_3_NODES)
                          .build(), "DECODER_H4")

                  .addLayer("DECODER_H2", new DenseLayer.Builder()
                          .nIn(DECODER_3_NODES)
                          .nOut(DECODER_2_NODES)
                          .build(), "DECODER_H3")

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

     public static void train(int epoch, ArrayList<INDArray> training_set) throws Exception
     {
          INDArray[] INPs = new INDArray[1];
          System.out.println("Beginning training...");
          for( ; epoch>0; epoch--)
          {
               for( INDArray t: training_set)
               {
                    INPs[0] = t;
                    model.fit(INPs, INPs);
               }
               System.out.println( model.score() + "\t" + epoch + " to go!");
          }
          model.save(new File("ae.zip"));
     }

     public static void test(ArrayList<INDArray> testing_set) throws Exception {
          model = ComputationGraph.load(new File("ae.zip"), false);
          INDArray[] INPs = new INDArray[1];
          System.out.println("Beginning testing...");

          Map<String,INDArray> activations = model.feedForward(INPs,false);

     }
}
