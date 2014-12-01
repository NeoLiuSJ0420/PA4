package deep;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U,b1,b2;
	protected SimpleMatrix dL,dW,dU,db1,db2;
	//
	public int windowSize,wordSize, hiddenSize;
	public double learningRate;
	public double lamda=0.0001;
	public int epoch=10;
	public String[] cat={"O","LOC","MISC","ORG","PER"};

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		windowSize=_windowSize;
		hiddenSize=_hiddenSize;
		learningRate=_lr;
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		double init=Math.sqrt(6.0)/Math.sqrt(50*windowSize+hiddenSize);
		W=SimpleMatrix.random(hiddenSize,windowSize*50,(-1)*init,init, new Random());
		b1=new SimpleMatrix(hiddenSize,1);
		U=SimpleMatrix.random(5,hiddenSize,(-1)*init,init, new Random());
		b2=new SimpleMatrix(5,1);
	}

	private SimpleMatrix extractL(List<Datum> Data,int count,int[] num) {
		SimpleMatrix ans=new SimpleMatrix(50*windowSize,1);
		String word=Data.get(count).word.toLowerCase();
		
		if (!FeatureFactory.wordToNum.containsKey(word))
			num[1]=0;
		else num[1]=FeatureFactory.wordToNum.get(word);
		
		
		if ((count==Data.size()-1)||(Data.get(count+1).word.toLowerCase().equals("-docstart-"))) 
			num[2]=31;
		else if (!FeatureFactory.wordToNum.containsKey(Data.get(count+1).word.toLowerCase()))
			num[2]=0;
		else
			num[2]=FeatureFactory.wordToNum.get(Data.get(count+1).word.toLowerCase());
		
		if ((count==0)||(Data.get(count-1).word.toLowerCase().equals("-docstart-"))) 
			num[0]=30;
		else if (!FeatureFactory.wordToNum.containsKey(Data.get(count-1).word.toLowerCase()))
			num[0]=0;
		else
			num[0]=FeatureFactory.wordToNum.get(Data.get(count-1).word.toLowerCase());
		
		for (int i=0;i<windowSize;i++) 
			ans=ans.combine(i*50, 0, NER.allVecs.extractVector(false,num[i]));
		return ans;
		
	}
	
	private double isSame(int a,int b) {
		if (a==b) return 1;
		return 0;
	}
	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> data ){
		int[] label=new int[data.size()];
		double score;
		for (int step=0;step<epoch;step++) {
			long seed = System.nanoTime();
			Collections.shuffle(data, new Random(seed));
			for (int i=0;i<data.size();i++) {
				if (data.get(i).label.equals("O")) label[i]=0;
				if (data.get(i).label.equals("LOC")) label[i]=1;
				if (data.get(i).label.equals("MISC")) label[i]=2;
				if (data.get(i).label.equals("ORG")) label[i]=3;
				if (data.get(i).label.equals("PER")) label[i]=4;
			}
			score=0;
			System.out.println(step);
			for (int count=0;count<data.size();count++) {
				String word=data.get(count).word.toLowerCase();
				SimpleMatrix y=new SimpleMatrix(5,1);//y,5*1
				y.set(label[count],0,1);
				if (word.equals("-docstart-")) continue;
				int[] num=new int[3];
				L=extractL(data,count,num);
				SimpleMatrix z=W.mult(L).plus(b1);//z,h*1
				SimpleMatrix a=new SimpleMatrix(z);//a,h*1
				for (int i=0;i<hiddenSize;i++)
					a.set(i,0,Math.tanh(a.get(i,0)));
				SimpleMatrix q=U.mult(a).plus(b2);//q,5*1
				double sum=0;
				for (int i=0;i<5;i++) sum+=Math.exp(q.get(i,0));
				SimpleMatrix p=new SimpleMatrix(5,1);//p,5*1
				for (int i=0;i<5;i++) p.set(i,0,Math.exp(q.get(i,0))/sum);
				
				//delta2 and delta1
				SimpleMatrix delta2=p.minus(y);//delta2,5*1
				SimpleMatrix delta1=new SimpleMatrix(hiddenSize,1);//delta1,h*1
				for (int i=0;i<hiddenSize;i++) {
					double temp=0;
					for (int j=0;j<5;j++) 
						temp+=delta2.get(j)*U.get(j,i)*(1-Math.pow(a.get(i), 2));
					delta1.set(i,0,temp);
				}
				//calculate gradients
				BackPropagation(delta2,delta1,a);
				//update
				U=U.minus(dU.scale(learningRate));
				b2=b2.minus(db2.scale(learningRate));
				W=W.minus(dW.scale(learningRate));
				b1=b1.minus(db1.scale(learningRate));
				for (int i=0;i<windowSize*50;i++) {
					int row=i%50;
					int col=num[i/50];
					NER.allVecs.set(row,col,NER.allVecs.get(row,col)-learningRate*dL.get(i,0));
				}
			}//finish 1 iter of SGD
	
			
		}
			
			
	
	}

	
	private void BackPropagation(SimpleMatrix delta2, SimpleMatrix delta1, SimpleMatrix a) {
		dU=delta2.mult(a.transpose()).plus(U.scale(lamda));
		db2=new SimpleMatrix(delta2);
		dW=delta1.mult(L.transpose()).plus(W.scale(lamda));
		db1=new SimpleMatrix(delta1);
		dL=W.transpose().mult(delta1);
	}

	public void test(List<Datum> testData) throws IOException{
		int[] label=new int[testData.size()];
		for (int i=0;i<testData.size();i++) {
			if (testData.get(i).label.equals("O")) label[i]=0;
			if (testData.get(i).label.equals("LOC")) label[i]=1;
			if (testData.get(i).label.equals("MISC")) label[i]=2;
			if (testData.get(i).label.equals("ORG")) label[i]=3;
			if (testData.get(i).label.equals("PER")) label[i]=4;
		}
		StringBuilder sb=new StringBuilder();
		for (int count=0;count<testData.size();count++) {
			String word=testData.get(count).word.toLowerCase();
			int y=label[count];
			if (word.equals("-docstart-")) continue;
			int[] num=new int[3];
			L=extractL(testData,count,num);
			SimpleMatrix z=W.mult(L).plus(b1);//z,h*1
			SimpleMatrix a=new SimpleMatrix(z);//a,h*1
			for (int i=0;i<hiddenSize;i++)
				a.set(i,0,Math.tanh(a.get(i,0)));
			SimpleMatrix q=U.mult(a).plus(b2);//q,5*1
			double sum=0;
			for (int i=0;i<5;i++) sum+=Math.exp(q.get(i,0));
			SimpleMatrix p=new SimpleMatrix(5,1);//p,5*1
			for (int i=0;i<5;i++) p.set(i,0,Math.exp(q.get(i,0))/sum);
			System.out.println(p);
			double max=0;int category=-1;
			for (int i=0;i<5;i++) {
				if (p.get(i,0)>max) {
					max=p.get(i,0);
					category=i;
				}
			}
			sb.append(String.format("%s	%s	%s\n",testData.get(count).word,testData.get(count).label,cat[category]));
		}
		PrintWriter writer = new PrintWriter("windowmodel.out", "UTF-8");
		writer.print(sb);
		writer.close();
		System.out.println("finish");
	}
	
	
}
