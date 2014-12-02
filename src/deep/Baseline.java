package deep;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

public class Baseline {
	public String[] cat={"O","LOC","MISC","ORG","PER"};
	
	public static void main(String[] args) throws IOException {
	    /*
		if (args.length < 2) {
		    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
		    return;
		}	    
		*/
		StringBuilder sb=new StringBuilder();
		// this reads in the train and test datasets
		List<Datum> trainData = FeatureFactory.readTrainData("data/train");
		List<Datum> testData = FeatureFactory.readTestData("data/test");
		HashMap<String,HashMap<String,Integer>> wordToTag=new HashMap<String,HashMap<String,Integer>>();
		for (Datum d:trainData) {
			if (!wordToTag.containsKey(d.word)) {
				HashMap<String,Integer> temp=new HashMap<String,Integer>();
				temp.put(d.label, 1);
				wordToTag.put(d.word, temp);
			}
			else {
				HashMap<String,Integer> temp=wordToTag.get(d.word);
				if (temp.containsKey(d.label)) {
					int count=temp.get(d.label);
					temp.put(d.label, count+1);
				}
				else temp.put(d.label,1);
			}
		}
		
		for (Datum d:testData) {
			if (wordToTag.containsKey(d.word)) {
				String ans="";int max=0;
				HashMap<String,Integer> temp=wordToTag.get(d.word);
				for (String str:temp.keySet()) {
					if (temp.get(str)>max) {
						max=temp.get(str);
						ans=str;
					}
				}
				sb.append(String.format("%s	%s	%s\n",d.word,d.label,ans));
			}
			else sb.append(String.format("%s	%s	%s\n",d.word,d.label,"O"));
			
		}
		PrintWriter writer = new PrintWriter("baseline.out", "UTF-8");
		writer.print(sb);
		writer.close();

		
		
	}

}
