package deep;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

public class Baseline {
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
		List<Datum> testData = FeatureFactory.readTestData("data/dev");
		HashMap<String,String> wordToTag=new HashMap<String,String>();
		for (Datum d:trainData) {
			if (!wordToTag.containsKey(d.word))
				wordToTag.put(d.word, d.label);
			else if (!d.label.equals("O"))
				wordToTag.put(d.word, d.label);
				
		}
		
		for (Datum d:testData) {
			if (wordToTag.containsKey(d.word)) {
				sb.append(String.format("%s	%s	%s\n",d.word,d.label,wordToTag.get(d.word)));
			}
			else {
				sb.append(String.format("%s	%s	O\n",d.word,d.label));
			}
			
		}
		PrintWriter writer = new PrintWriter("baseline.out", "UTF-8");
		writer.print(sb);
		writer.close();

		
		
	}

}
