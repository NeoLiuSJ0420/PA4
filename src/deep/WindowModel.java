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
	public int epoch=20;
	public String[] cat={"O","LOC","MISC","ORG","PER"};
	public String[] days={"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","January","February","March","April","May","June","July","August","September","October","November","December"};
	public HashSet<String> specialStrings;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		windowSize=_windowSize;
		hiddenSize=_hiddenSize;
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
		W=SimpleMatrix.random(hiddenSize,windowSize*50,(-1)*init,init, new Random());
		b1=new SimpleMatrix(hiddenSize,1);
		U=SimpleMatrix.random(5,hiddenSize,(-1)*init,init, new Random());
		b2=new SimpleMatrix(5,1);
	}
    public double feedForward(SimpleMatrix y) {
       SimpleMatrix z=W.mult(L).plus(b1);//z,h*1
	   SimpleMatrix a=new SimpleMatrix(z);//a,h*1
	   for (int i=0;i<hiddenSize;i++)
		   a.set(i,0,Math.tanh(a.get(i,0)));
	   SimpleMatrix q=U.mult(a).plus(b2);//q,5*1
	   double sum=0;
	   for (int i=0;i<5;i++) sum+=Math.exp(q.get(i,0));
	       SimpleMatrix p=new SimpleMatrix(5,1);//p,5*1
           for (int i=0;i<5;i++) p.set(i,0,Math.exp(q.get(i,0))/sum);
           double J = 0;
           for(int i=0;i<5;i++) J+=Math.log(p.get(i,0)) * y.get(i,0);
           J = J*-1;
           return J;
    }
       public void doGradCheck(SimpleMatrix y){
          double eps = 1e-4;
          double diff = 0.0;
          double t;
          double J_plus, J_minus,grad,graddiff;
          for(int i = 0; i<L.getNumElements(); i++){
            t = L.get(i);
            L.set(i,t+eps);
            J_plus = feedForward(y);
            L.set(i,t-eps);
            J_minus = feedForward(y);
            grad = (J_plus-J_minus) / (2*eps);
            graddiff = Math.abs(dL.get(i) - grad);
           // System.out.println("dL="+dL.get(i)+"  numgrad="+grad);
            diff += graddiff*graddiff;
            L.set(i,t);
          }
          for(int i = 0; i< W.getNumElements();i++){
            t = W.get(i);
            W.set(i,t+eps);
            J_plus = feedForward(y);
            W.set(i,t-eps);
            J_minus = feedForward(y);
            grad = (J_plus-J_minus) /(2*eps);
            graddiff = Math.abs(dW.get(i)-grad);
           // System.out.println("dW="+dW.get(i)+"  numgrad="+grad);
            diff += graddiff*graddiff;
            W.set(i,t);
         } 
         for(int i = 0; i < U.getNumElements();i++){
           t = U.get(i);
           U.set(i,t+eps);
           J_plus = feedForward(y);
           U.set(i,t-eps);
           J_minus = feedForward(y);
           grad = (J_plus - J_minus)/(2*eps);
           graddiff = Math.abs(dU.get(i) -grad);
           //System.out.println("dU="+dU.get(i)+"   numgrad="+grad);
           diff+= graddiff*graddiff;
           U.set(i,t);
         }
         for(int i = 0; i< b1.getNumElements();i++){
           t = b1.get(i);
           b1.set(i,t+eps);
           J_plus = feedForward(y);
           b1.set(i,t-eps);
           J_minus = feedForward(y);
           grad = (J_plus - J_minus)/(2*eps);
           graddiff = Math.abs(db1.get(i)-grad);
           diff+= graddiff*graddiff;
           b1.set(i,t);
         }
          for(int i =0;i<b2.getNumElements();i++) {
           t = b2.get(i);
           b2.set(i,t+eps);
           J_plus = feedForward(y);
           b2.set(i,t-eps);
           J_minus = feedForward(y);
           grad = (J_plus - J_minus)/(2*eps);
           graddiff = Math.abs(db2.get(i)-grad);
           diff+= graddiff*graddiff;
           b2.set(i,t);
         }
           diff = Math.sqrt(diff);
           //System.out.println("diff="+diff);
           if(diff>1e-7) System.out.println("fail gradient check");
           
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
			score=0;
			j=Math.pow(W.normF(),2)+Math.pow(U.normF(),2);
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
				score+=Math.log(p.get(label[count],0));
				
				//delta2 and delta1
				SimpleMatrix delta2=p.minus(y);//delta2,5*1
				SimpleMatrix delta1=new SimpleMatrix(hiddenSize,1);//delta1,h*1
				for (int i=0;i<hiddenSize;i++) {
					double temp=0;
					for (int k=0;k<5;k++) 
						temp+=delta2.get(k)*U.get(k,i)*(1-Math.pow(a.get(i), 2));
					delta1.set(i,0,temp);
				}
				//calculate gradients
				BackPropagation(delta2,delta1,a);
				

				//if(step==0&&count<=10) doGradCheck(y);

				//regularize
                           
                dU = dU.plus(U.scale(lamda));
                dW = dW.plus(W.scale(lamda));
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
			score=score*(-1)/data.size();
			j=j*lamda/(2*data.size())+score;
			System.out.println(j);
		}
			
			
	
	}

	
	private void BackPropagation(SimpleMatrix delta2, SimpleMatrix delta1, SimpleMatrix a) {
		dU=delta2.mult(a.transpose());
		db2=new SimpleMatrix(delta2);
		dW=delta1.mult(L.transpose());
		db1=new SimpleMatrix(delta1);
		dL=W.transpose().mult(delta1);
	}

	public void test(List<Datum> testData,String data) throws IOException{
		PrintWriter writer = new PrintWriter(data+"_windowmodel.out", "UTF-8");
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
			SimpleMatrix z=W.mult(L).plus(b1);//z,h*1
			SimpleMatrix a=new SimpleMatrix(z);//a,h*1
			for (int i=0;i<hiddenSize;i++)
				a.set(i,0,Math.tanh(a.get(i,0)));
			SimpleMatrix q=U.mult(a).plus(b2);//q,5*1
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
