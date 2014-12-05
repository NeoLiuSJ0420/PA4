package deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
	static SimpleMatrix allVecs;
    public static void main(String[] args) throws IOException {
    /*
	if (args.length < 2) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
	    return;
	}	    
	*/

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData("data/train");
	
	
	//	read the train and test data
	//TODO: Implement this function (just reads in vocab and word vectors)
	FeatureFactory.initializeVocab("data/vocab.txt");
	allVecs= FeatureFactory.readWordVectors("data/wordVectors.txt");
	// initialize model 
	WindowModel model = new WindowModel(3,20,0.001);
	model.initWeights();

	//TODO: Implement those two functions
	
	model.train(trainData);
	List<Datum> testData_train = FeatureFactory.readTestData("data/train");
	model.test(testData_train,"train");
	List<Datum> testData_dev = FeatureFactory.readTestData("data/dev");
	model.test(testData_dev,"dev");
	List<Datum> testData_test = FeatureFactory.readTestData("data/test");
	model.test(testData_test,"test");
    }
}