package deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U,b1,b2;
	//
	public int windowSize,wordSize, hiddenSize;
	public double learningRate;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		windowSize=_windowSize;
		hiddenSize=_hiddenSize;
		learningRate=_lr;
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		double init=Math.sqrt(6)/Math.sqrt(50*windowSize+hiddenSize);
		W=SimpleMatrix.random(hiddenSize,windowSize*50,(-1)*init,init, new Random());
		b1=SimpleMatrix.random(hiddenSize,1,(-1)*init,init, new Random());
		U=SimpleMatrix.random(5,hiddenSize,(-1)*init,init, new Random());
		b2=SimpleMatrix.random(5,1,(-1)*init,init, new Random());
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		int[] num=new int[3];
		for (int count=0;count<_trainData.size();count++) {
			String word=_trainData.get(count).word.toLowerCase();
			if (word.equals("-docstart-")) continue;
			num[1]=FeatureFactory.wordToNum.get(word);
			
			if ((count==_trainData.size()-1)||(_trainData.get(count+1).word.toLowerCase().equals("-DOCSTART-"))) 
				num[2]=31;
			else
				num[2]=FeatureFactory.wordToNum.get(_trainData.get(count+1).word.toLowerCase());
			//System.out.println(count);
			if ((count==0)||(_trainData.get(count-1).word.toLowerCase().equals("-docstart-"))) 
				num[0]=30;
			else
				num[0]=FeatureFactory.wordToNum.get(_trainData.get(count-1).word.toLowerCase());
			L=new SimpleMatrix(50*windowSize,1);
			for (int i=0;i<windowSize;i++) 
				L=L.combine(i*50, 0, NER.allVecs.extractVector(false,num[i]));
			
			SimpleMatrix z=W.mult(L).plus(b1);
			
			
		}
			
			
	
	}

	
	public void test(List<Datum> testData){
		// TODO
		}
	
}
