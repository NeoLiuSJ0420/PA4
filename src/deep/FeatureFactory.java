package deep;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.*;

import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
	    // TODO: you'd want to handle sentence boundaries
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		LineNumberReader  lnr = new LineNumberReader(new FileReader(new File(vecFilename)));
		lnr.skip(Long.MAX_VALUE);
		int size=lnr.getLineNumber();
		// Finally, the LineNumberReader object should be closed to prevent resource leak
		lnr.close();
		double[][] vecData=new double[50][size];
		BufferedReader in = new BufferedReader(new FileReader(vecFilename));
		for (int i=0;i<size;i++) {
			String line=in.readLine();
			StringTokenizer st=new StringTokenizer(line);
			for (int j=0;j<50;j++) {
				vecData[j][i]=Double.parseDouble(st.nextToken());
			}
		}
		allVecs=new SimpleMatrix(vecData);
		
		if (allVecs!=null) return allVecs;
		return null;
		//TODO implement this
		//set allVecs from filename		

	}
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
		int count=0;
		BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			wordToNum.put(line.toLowerCase(), count);
			numToWord.put(count, line.toLowerCase());
			count++;
		}
		return wordToNum;
	}
 








}
