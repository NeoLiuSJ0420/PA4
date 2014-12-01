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
		U=SimpleMatrix.random(hiddenSize,5,(-1)*init,init, new Random());
		b2=new SimpleMatrix(5,1);
	}

	private SimpleMatrix extractL(List<Datum> Data,int count,int[] num) {
		SimpleMatrix ans=new SimpleMatrix(50*windowSize,1);
		String word=Data.get(count).word.toLowerCase();
		num[1]=FeatureFactory.wordToNum.get(word);
		
		
		if ((count==Data.size()-1)||(Data.get(count+1).word.toLowerCase().equals("-docstart-"))) 
			num[2]=31;
		else
			num[2]=FeatureFactory.wordToNum.get(Data.get(count+1).word.toLowerCase());
		if ((count==0)||(Data.get(count-1).word.toLowerCase().equals("-docstart-"))) 
			num[0]=30;
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
				int y=label[count];
				if (word.equals("-docstart-")) continue;
				if (count!=0)
					if (!FeatureFactory.wordToNum.containsKey(data.get(count-1).word.toLowerCase())) 
						if (!data.get(count-1).word.toLowerCase().equals("-docstart-"))
							continue;
				if (!FeatureFactory.wordToNum.containsKey(data.get(count).word.toLowerCase())) continue;
				if (count!=data.size()-1)
					if (!FeatureFactory.wordToNum.containsKey(data.get(count+1).word.toLowerCase())) 
						if (!data.get(count+1).word.toLowerCase().equals("-docstart-"))
							continue;
				int[] num=new int[3];
				L=extractL(data,count,num);
				SimpleMatrix z=W.mult(L).plus(b1);//z,h*1
				SimpleMatrix a=new SimpleMatrix(z);//a,h*1
				for (int i=0;i<hiddenSize;i++)
					a.set(i,0,Math.tanh(a.get(i,0)));
				SimpleMatrix q=U.transpose().mult(a).plus(b2);//q,5*1
				double sum=0;
				for (int i=0;i<5;i++) sum+=Math.exp(q.get(i,0));
				SimpleMatrix p=new SimpleMatrix(5,1);//p,5*1
				for (int i=0;i<5;i++) p.set(i,0,Math.exp(q.get(i,0))/sum);
				score+=Math.log(p.get(y,0));
				
				double[] delta1=new double[5];
				for (int i=0;i<5;i++)
					delta1[i]=(-1)*(isSame(i,y)-p.get(i));
				//update U
				for (int i=0;i<hiddenSize;i++)
					for (int j=0;j<5;j++) {
						double temp=U.get(i,j);
						double gradient=delta1[j]*a.get(i)+lamda*temp;
						U.set(i,j,temp-learningRate*gradient);
					}
				//update b2
				for (int j=0;j<5;j++) {
					double temp=b2.get(j,0);
					double gradient=delta1[j];
					b2.set(j,0,temp-learningRate*gradient);
				}
				//update W
				for (int k=0;k<hiddenSize;k++)
					for (int j=0;j<50*windowSize;j++) {
						double temp=W.get(k,j);
						double gradient=0;
						for (int i=0;i<5;i++) {
							gradient+=delta1[i]*U.get(k,i)*(1-Math.pow(a.get(k,0),2))*L.get(j)+lamda*temp;
						}
						W.set(k,j,temp-learningRate*gradient);
					}
				//update b1
				for (int k=0;k<hiddenSize;k++) {
					double temp=b1.get(k,0);
					double gradient=0;
					for (int i=0;i<5;i++) {
						gradient+=delta1[i]*U.get(k,i)*(1-Math.pow(a.get(k,0),2));
					}
					b1.set(k,0,temp-learningRate*gradient);
				}
				//update allVecs/L
				for (int l=0;l<windowSize*50;l++) {
					int row=l%windowSize;
					int col=num[l/50];
					double temp=NER.allVecs.get(row,col);
					double gradient=0;
					for (int i=0;i<5;i++) 
						for (int j=0;j<hiddenSize;j++) {
							gradient+=delta1[i]*U.get(j,i)*(1-Math.pow(a.get(j,0),2))*W.get(j,l);
						}
					NER.allVecs.set(row,col,temp-learningRate*gradient);
				}
			}//finish 1 iter of SGD
	
			
		}
			
			
	
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
			if (count!=0)
				if (!FeatureFactory.wordToNum.containsKey(testData.get(count-1).word.toLowerCase())) 
					if (!testData.get(count-1).word.toLowerCase().equals("-docstart-"))
						continue;
			if (!FeatureFactory.wordToNum.containsKey(testData.get(count).word.toLowerCase())) continue;
			if (count!=testData.size()-1)
				if (!FeatureFactory.wordToNum.containsKey(testData.get(count+1).word.toLowerCase())) 
					if (!testData.get(count+1).word.toLowerCase().equals("-docstart-"))
						continue;
			int[] num=new int[3];
			L=extractL(testData,count,num);
			SimpleMatrix z=W.mult(L).plus(b1);//z,h*1
			SimpleMatrix a=new SimpleMatrix(z);//a,h*1
			for (int i=0;i<hiddenSize;i++)
				a.set(i,0,Math.tanh(a.get(i,0)));
			SimpleMatrix q=U.transpose().mult(a).plus(b2);//q,5*1
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
