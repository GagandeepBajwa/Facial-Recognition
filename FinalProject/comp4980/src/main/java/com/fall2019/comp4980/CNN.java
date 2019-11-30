package com.fall2019.comp4980;
import org.bytedeco.javacv.FrameFilter;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.deeplearning4j.earlystopping.scorecalc.AutoencoderScoreCalculator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.xml.crypto.Data;
import java.io.File;
import java.util.*;

public class CNN {

    static ComputationGraph model;
    static final int VECTOR_INPUT = 7500;
    static final int INPUT_NODES = 7500;
    static final int CONV_2_NODES = 16;
    static final int INPUT_CHANNELS =1;
    static final int CONV_2_OUTPUTS = 32;
    static final int HIDDEN_H2_NODES = 250;
    static final int HIDDEN_H3_NODES = 125;
    static final int HIDDEN_H4_NODES = 60;
    static final int HIDDEN_H5_NODES = 30;
    static final int OUTPUT_NODES = 11;
    static final int OUTPUTS = 11;


    CNN(double learningRate, boolean newModel)throws Exception {

        if(newModel){
            model = nn_init(learningRate);
        }
        else{
            model = ComputationGraph.load(new File("cnn/cnn_9625_1.5164661737349599E-4.zip"), true);
            // cnn_9625_9.31903758214503E-6.zip
            model.setLearningRate(learningRate);
        }

    }
    private static ComputationGraph nn_init(double learningRate)
    {

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.ELU)
                .updater(new Adam(learningRate))

                .graphBuilder()

                /* Begin. Start creating the neural network structure (Layers) here */

                .addInputs("vector_in")             // Name of the layer(s) for the inputs

                .addLayer("INPUT_I1", new ConvolutionLayer.Builder()  // First layer of type Convolutional, we name it INPUT_I1
                        .kernelSize(3,3)    // Default receptive field of 2,2
                        .stride(3,3)
                        .nIn(INPUT_CHANNELS)             // 3 input channels (red,green,blue) to start
                        .nOut(CONV_2_NODES)            // We want 8 feature maps
                        .build(), "vector_in")

                .addLayer("SUB1", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(3,3)    // Standard pool kernel of 2x2. Keep the largest value (max pooling).
                        .stride(3,3)        // Stride should be the same as kernel size so the kernels don't overlap.
                        .build(), "INPUT_I1")


                /** Keep adding convolutional/pooling layers as many as you want */

                .addLayer("CONV2", new ConvolutionLayer.Builder()
                        .kernelSize(1,1)
                        .stride(1,1)
                        .nIn(CONV_2_NODES)
                        .nOut(CONV_2_OUTPUTS)
                        .build(), "SUB1")

                .addLayer("SUB2", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(3,3)
                        .stride(3,3)
                        .build(), "CONV2")

                /** Here is where we transition from convolutional to standard feed forward network. Note that
                 for this transition you don't need to specify an nIn(). DeepLearning4j will figure this out for
                 you to automatically connect the convolution part of the network to the feed forward */

                .addLayer("DENSE1", new DenseLayer.Builder()     /** Standard feed forward from here */
                        .nOut(HIDDEN_H2_NODES)
                        .build(), "SUB2")


                .addLayer("HIDDEN_H2", new DenseLayer.Builder()
                        .nIn(HIDDEN_H2_NODES)
                        .nOut(HIDDEN_H3_NODES)
                        .build(), "DENSE1")

                .addLayer("HIDDEN_H3", new DenseLayer.Builder()
                        .nIn(HIDDEN_H3_NODES)
                        .nOut(HIDDEN_H4_NODES)
                        .build(), "HIDDEN_H2")


                .addLayer("HIDDEN_H4", new DenseLayer.Builder()
                        .nIn(HIDDEN_H4_NODES)
                        .nOut(HIDDEN_H5_NODES)
                        .build(), "HIDDEN_H3")

                .addLayer("HIDDEN_H5", new DenseLayer.Builder()
                        .nIn(HIDDEN_H5_NODES)
                        .nOut(OUTPUT_NODES)
                        .build(), "HIDDEN_H4")

                .addLayer("OUTPUT_O1", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation( Activation.SOFTMAX )
                        .nIn( OUTPUT_NODES )
                        .nOut( OUTPUTS )               // we have 10 outputs
                        .build(), "HIDDEN_H5")

                .setOutputs("OUTPUT_O1")

                /** Important: you must tell the CNN the height,width and depth(channels) of the images
                 * you are going to use. In this example the square that slides across the GUI looking for
                 * Johnny is 128x94 height x width and because we are using RGB the channels are 3.
                 */

                .setInputTypes(InputType.convolutional(28, 28, 3))    // height,width,channels


                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();
        return net;
    }

    public static void train2(int epoch, ArrayList<INDArray> training_set, Dataset ds) throws Exception{
        
        INDArray[] INPs = new INDArray[1];
        INDArray[] OUTs = new INDArray[1];
        System.out.println("Beginning training...");
        int person_identifier = 0;
        int no_data_person = 10;

        Map<String, ArrayList<INDArray>> trainMap = ds.trainMap;
        for( ; epoch>0; epoch--)
        {
            double score = 0.0;
            int counter = 0;
            person_identifier=0;
            for( String person : trainMap.keySet())
            {
                double batchscore = 0.0;
                ArrayList<INDArray> personTrainExamples = trainMap.get(person);
                for(INDArray personTrain : personTrainExamples){

                    OUTs[0] = Nd4j.zeros(new int[]{1, 11});
                    OUTs[0].putScalar(person_identifier, 1.0);
                    //System.out.println(person +"  "+ person_identifier);

                    INPs[0] = personTrain;
                    model.fit(INPs, OUTs);
                    counter++;
                    score += model.score();
                    batchscore += model.score();
                }
                person_identifier++;
                //System.out.println("Person score:" + batchscore/10);
            }
            System.out.println( score/counter + "\t" + epoch + " to go!");
            if(epoch%25==0){model.save(new File("cnn" + "_" + epoch + "_"+ score/counter + ".zip"), true);}
        }
        model.save(new File("cnn.zip"));
    }
    
    
    public static void test(Dataset ds) throws Exception {
        //model = ComputationGraph.load(new File("cnn.zip"), false);

        System.out.println("Beginning testing...");

        int false_rejections = 0;
        int false_acceptaces = 0;
        int positive_acceptances = 0;

        int person_identifier =0;
        for(String person : ds.testMap.keySet()) {
            System.out.println("");
            System.out.println("");
            System.out.println("PERSON: " + person);


            /**Get score for each person when running on testing set of the same person*/
            for (INDArray personExample : ds.testMap.get(person)) {
                INDArray[] res = model.output(personExample);
                System.out.println(res[0]);
                int result = Nd4j.getExecutioner().exec(new IAMax(res[0])).getInt();
                System.out.println("Maximum value and it index is: " + res[0].max() + "and indexof it is: " + result);//+ result.indexOf(Collections.max(result.get(0))));
                if (result == person_identifier) {
                    positive_acceptances++;
                } else {
                    false_rejections++;
                }
            }
            System.out.println("Positive Acceptances: " + positive_acceptances);
            System.out.println("False Rejections: " + false_rejections);


//            for(String person2:ds.testMap.keySet()){
//                if(person.compareTo(person2)!=0){
//                    System.out.println("Comparing with: " + person2);
//                    ArrayList<INDArray> others = ds.testMap.get(person2);
//                    for(INDArray other : others){
//                        INDArray[] res = model.output( other );
//                        System.out.println(res[0]);
//                        List result = new ArrayList(Arrays.asList(res[0]));
//                        if(result.indexOf(Collections.max(result))==person_identifier){
//                            false_acceptaces++;
//                        }
//                    }
//        }

            System.out.println("False Acceptance: " + false_acceptaces);

            person_identifier++;
            false_acceptaces = 0;
            false_rejections = 0;
            positive_acceptances = 0;

        }

        /** This demo takes a lot of memory in forcing some kind of garbage collection seems to help (eliminated out-of-memory errors) */
        java.lang.Runtime.getRuntime().gc();
    }


}
