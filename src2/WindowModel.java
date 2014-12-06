package deep;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W1,W2, U,b1,b2,b3;
	protected SimpleMatrix dL,dW1,dW2,dU,db1,db2,db3;
	//
	public int windowSize,wordSize, hiddenSize, hiddenSize2 = 100;
	public double learningRate;
	public double lamda=0.0001;
	public int epoch=20;
	public String[] cat={"O","LOC","MISC","ORG","PER"};
	public String[] days={"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","January","February","March","April","May","June","July","August","September","October","November","December"};
	public HashSet<String> specialStrings;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		windowSize=_windowSize;
		hiddenSize= 150;
		learningRate=_lr;
		specialStrings=new HashSet<String>();
		for (String str:days)
			specialStrings.add(str);
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		double init=Math.sqrt(6.0)/Math.sqrt(50*windowSize+hiddenSize);
		W1=SimpleMatrix.random(hiddenSize,windowSize*50,(-1)*init,init, new Random());
		b1=new SimpleMatrix(hiddenSize,1);
                init = Math.sqrt(6.0)/Math.sqrt(hiddenSize+hiddenSize2);
                W2=SimpleMatrix.random(hiddenSize2,hiddenSize,(-1)*init,init, new Random());
                b2=new SimpleMatrix(hiddenSize2,1);
		U=SimpleMatrix.random(5,hiddenSize2,(-1)*init,init, new Random());
		b3=new SimpleMatrix(5,1);
	}
   
	private SimpleMatrix extractL(List<Datum> Data,int count,int[] num) {
		SimpleMatrix ans=new SimpleMatrix(50*windowSize,1);
		String word=Data.get(count).word.toLowerCase();
		
		if (!FeatureFactory.wordToNum.containsKey(word))
			num[1]=0;
		else num[1]=FeatureFactory.wordToNum.get(word);
		
		
		if ((count==Data.size()-1)||(Data.get(count+1).word.toLowerCase().equals("-docstart-")||(word.equals("."))))
			num[2]=31;
		else if (!FeatureFactory.wordToNum.containsKey(Data.get(count+1).word.toLowerCase()))
			num[2]=0;
		else
			num[2]=FeatureFactory.wordToNum.get(Data.get(count+1).word.toLowerCase());
		
		if ((count==0)||(Data.get(count-1).word.toLowerCase().equals("-docstart-"))||(Data.get(count-1).word.toLowerCase().equals("."))) 
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
	 * SGD training 
	 */
	public void train(List<Datum> data ){
		int[] label=new int[data.size()];
		double score;
		double j;
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
	                System.out.println("step="+step);
			for (int count=0;count<data.size();count++) {
				String word=data.get(count).word.toLowerCase();
				SimpleMatrix y=new SimpleMatrix(5,1);//y,5*1
				y.set(label[count],0,1);
				if (word.equals("-docstart-")) continue;
				int[] num=new int[3];
				L=extractL(data,count,num);
				SimpleMatrix z1=W1.mult(L).plus(b1);
				SimpleMatrix a1=new SimpleMatrix(z1);
				for (int i=0;i<hiddenSize;i++)
					a1.set(i,0,Math.tanh(a1.get(i,0)));
                                SimpleMatrix z2=W2.mult(a1).plus(b2);
                                SimpleMatrix a2=new SimpleMatrix(z2);
                                for (int i=0;i<hiddenSize2;i++)
                                        a2.set(i,0,Math.tanh(a2.get(i,0)));
				SimpleMatrix q=U.mult(a2).plus(b3);
				double sum=0;
				for (int i=0;i<5;i++) sum+=Math.exp(q.get(i,0));
				SimpleMatrix p=new SimpleMatrix(5,1);
				for (int i=0;i<5;i++) p.set(i,0,Math.exp(q.get(i,0))/sum);
				
				//delta1, delta2, and delta3
				SimpleMatrix delta3=p.minus(y);
				SimpleMatrix delta2=new SimpleMatrix(hiddenSize2,1);
				for (int i=0;i<hiddenSize2;i++) {
					double temp=0;
					for (int k=0;k<5;k++) 
						temp+=delta3.get(k)*U.get(k,i)*(1-Math.pow(a2.get(i), 2));
					delta2.set(i,0,temp);
				}
                                SimpleMatrix delta1 = new SimpleMatrix(hiddenSize,1);
                                delta1 = W2.transpose().mult(delta2);
                                for (int i=0;i<hiddenSize;i++) {
                                   double temp;
                                   temp = delta1.get(i) * (1-Math.pow(a1.get(i),2));
                                   delta1.set(i,0,temp);
                                }
				//calculate gradients
				dU=delta3.mult(a2.transpose());
                                db3=new SimpleMatrix(delta3);
                                dW2=delta2.mult(a1.transpose());
                                db2=new SimpleMatrix(delta2);
                                dW1=delta1.mult(L.transpose());
                                db1=new SimpleMatrix(delta1);
                                dL=W1.transpose().mult(delta1);
				

				//regularize
                           
                                dU = dU.plus(U.scale(lamda));
                                dW1 = dW1.plus(W1.scale(lamda));
                                dW2 = dW2.plus(W2.scale(lamda));
                                //update
				U=U.minus(dU.scale(learningRate));
                                b3=b3.minus(db3.scale(learningRate));
                                W2=W2.minus(dW2.scale(learningRate));
				b2=b2.minus(db2.scale(learningRate));
				W1=W1.minus(dW1.scale(learningRate));
				b1=b1.minus(db1.scale(learningRate));


                                 

				for (int i=0;i<windowSize*50;i++) {
					int row=i%50;
					int col=num[i/50];
					NER.allVecs.set(row,col,NER.allVecs.get(row,col)-learningRate*dL.get(i,0));
				}
			}//finish 1 iter of SGD
		}
			
			
	
	}

	public void test(List<Datum> testData,String data) throws IOException{
		PrintWriter writer = new PrintWriter(data+"_windowmodeldeeper.out", "UTF-8");
		int[] label=new int[testData.size()];
		for (int i=0;i<testData.size();i++) {
			if (testData.get(i).label.equals("O")) label[i]=0;
			if (testData.get(i).label.equals("LOC")) label[i]=1;
			if (testData.get(i).label.equals("MISC")) label[i]=2;
			if (testData.get(i).label.equals("ORG")) label[i]=3;
			if (testData.get(i).label.equals("PER")) label[i]=4;
		}

		for (int count=0;count<testData.size();count++) {
			String word=testData.get(count).word.toLowerCase();
			String originalword=testData.get(count).word;
			int y=label[count];
			if (word.equals("-docstart-")) {
				writer.print(String.format("%s	%s	%s\n",testData.get(count).word,testData.get(count).label,"O"));
				continue;
			}
			int[] num=new int[3];
			L=extractL(testData,count,num);
			SimpleMatrix z1=W1.mult(L).plus(b1);
			SimpleMatrix a1=new SimpleMatrix(z1);
			for (int i=0;i<hiddenSize;i++)
				a1.set(i,0,Math.tanh(a1.get(i,0)));
                        SimpleMatrix z2=W2.mult(a1).plus(b2);
                        SimpleMatrix a2= new SimpleMatrix(z2);
                        for (int i=0;i<hiddenSize2;i++)
                              a2.set(i,0,Math.tanh(a2.get(i,0)));
			SimpleMatrix q=U.mult(a2).plus(b3);
			double sum=0;
			for (int i=0;i<5;i++) sum+=Math.exp(q.get(i,0));
			SimpleMatrix p=new SimpleMatrix(5,1);//p,5*1
			for (int i=0;i<5;i++) p.set(i,0,Math.exp(q.get(i,0))/sum);
			//System.out.println(p);
			double max=0;int category=-1;
			for (int i=0;i<5;i++) {
				if (p.get(i,0)>max) {
					max=p.get(i,0);
					category=i;
				}
			}
			if (word.equals(originalword))
				category=0;
			//handle cases
			if ((category==0)&&(originalword.length()>1)&&(Character.isUpperCase(originalword.charAt(0)))&&(!Character.isUpperCase(originalword.charAt(1)))) {
				if ((!Character.isLetter(testData.get(count-1).word.charAt(0)))&&(!Character.isDigit(testData.get(count-1).word.charAt(0))))
					continue;
				if (specialStrings.contains(originalword))
					continue;
				max=0;category=-1;
				for (int i=1;i<5;i++) {
					if (p.get(i,0)>max) {
						max=p.get(i,0);
						category=i;
					}
				}	
			}
			writer.print(String.format("%s	%s	%s\n",testData.get(count).word,testData.get(count).label,cat[category]));
		}
		
		writer.close();
		System.out.println("finish");
	}
	
	
}
