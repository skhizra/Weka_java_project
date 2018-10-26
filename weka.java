package project;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.Utils;
import weka.attributeSelection.*;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;




public class weka3 {

	public static void main(String[] args) throws Exception {
		 
		BufferedReader reader = new BufferedReader(
                 new FileReader("/Users/shufakhizra/Desktop/data mining/australian_outliers_removed_normalized.arff"));
Instances data1 = new Instances(reader);
reader.close();

data1.setClassIndex(data1.numAttributes() - 1);
int folds = 5;
AttributeSelection attsel = new AttributeSelection();  // package weka.attributeSelection!
CorrelationAttributeEval eval1 = new CorrelationAttributeEval();
//CorrelationAttributeEval eval2 = new CorrelationAttributeEval();
InfoGainAttributeEval infoeval = new InfoGainAttributeEval();

Ranker search = new Ranker();


Random rand = new Random(1);

data1.randomize(rand);
if (data1.classAttribute().isNominal())
  data1.stratify(folds);


System.out.println("Feature Selection: Correlation Classification Algorithm: Logistic Regresssion");


int[] arr = new int[10];
Evaluation eval = new Evaluation(data1);
for (int n = 0; n < folds; n++) {
	
	  Instances train = data1.trainCV(folds, n);
	  Instances test = data1.testCV(folds, n);

	attsel.setEvaluator(eval1);
	attsel.setSearch(search);
	attsel.SelectAttributes(train);
	//obtain the attribute indices that were selected
	int[] indices = attsel.selectedAttributes();
	for(int i=0; i<9; i++)
	{
		arr[i] = indices[i];
	}
	arr[9]= indices[(indices.length)-1];
	//System.out.println(indices.length);
	Remove remove = new Remove();
	remove.setAttributeIndicesArray(arr);
	//remove.setAttributeIndices(indices);
	remove.setInvertSelection(true);
	remove.setInputFormat(train);
	Instances train_data = Filter.useFilter(train, remove);
	
	remove.setInputFormat(test);
	Instances test_data = Filter.useFilter(test, remove);

	//System.out.println("Attributes selected are: " + Utils.arrayToString(arr));
	System.out.println("Attributes selected are: " + Utils.arrayToString(indices));

	

Logistic nb = new Logistic();
  nb.buildClassifier(train_data);
  eval.evaluateModel(nb, test_data);
  //System.out.println(eval.toSummaryString());

}

System.out.println(eval.toSummaryString(folds + "-fold Cross-validation Results", false));
System.out.println("Precision: " + eval.precision(1));
System.out.println("Recall : " + eval.recall(1));
System.out.println("FMeasure: " + eval.fMeasure(1));
System.out.println();


System.out.println("Feature Selection: Correlation ; Classification Algorithm: Random Forest");
int[] arr1 = new int[10];
Evaluation eval2 = new Evaluation(data1);
for (int n = 0; n < folds; n++) {
	
	Instances train = data1.trainCV(folds, n);
	Instances test = data1.testCV(folds, n);

	attsel.setEvaluator(eval1);
	attsel.setSearch(search);
	attsel.SelectAttributes(train);
	//obtain the attribute indices that were selected
	int[] indices = attsel.selectedAttributes();
	for(int i=0; i<9; i++)
	{
		arr1[i] = indices[i];
	}
	arr1[9]= indices[(indices.length)-1];
	//System.out.println(indices.length);
	Remove remove = new Remove();
	remove.setAttributeIndicesArray(arr1);
	//remove.setAttributeIndices(indices);
	remove.setInvertSelection(true);
	remove.setInputFormat(train);
	Instances train_data = Filter.useFilter(train, remove);
	
	remove.setInputFormat(test);
	Instances test_data = Filter.useFilter(test, remove);

	//System.out.println("Attributes selected are: " + Utils.arrayToString(arr1));
	System.out.println("Attributes selected are: " + Utils.arrayToString(indices));

	

  RandomForest nb = new RandomForest();
  nb.buildClassifier(train_data);
  eval2.evaluateModel(nb, test_data);
  //System.out.println(eval2.toSummaryString());

}

System.out.println(eval2.toSummaryString(folds + "-fold Cross-validation Results", false));
System.out.println("Precision: " + eval2.precision(1));
System.out.println("Recall : " + eval2.recall(1));
System.out.println("FMeasure: " + eval2.fMeasure(1));
System.out.println();


System.out.println("Feature Selection: Information Gain ; Classification Algorithm: Logistic Regresssion");



int[] arr2 = new int[10];
Evaluation eval3 = new Evaluation(data1);
for (int n = 0; n < folds; n++) {
	
	  Instances train = data1.trainCV(folds, n);
	  Instances test = data1.testCV(folds, n);

	attsel.setEvaluator(infoeval);
	attsel.setSearch(search);
	attsel.SelectAttributes(train);
	//obtain the attribute indices that were selected
	int[] indices = attsel.selectedAttributes();
	for(int i=0; i<9; i++)
	{
		arr2[i] = indices[i];
	}
	arr2[9]= indices[(indices.length)-1];
	//System.out.println(indices.length);
	Remove remove = new Remove();
	remove.setAttributeIndicesArray(arr2);
	//remove.setAttributeIndices(indices);
	remove.setInvertSelection(true);
	remove.setInputFormat(train);
	Instances train_data = Filter.useFilter(train, remove);
	
	remove.setInputFormat(test);
	Instances test_data = Filter.useFilter(test, remove);

	//System.out.println("Attributes selected are: " + Utils.arrayToString(arr2));
	System.out.println("Attributes selected are: " + Utils.arrayToString(indices));

	

Logistic nb = new Logistic();
  nb.buildClassifier(train_data);
  eval3.evaluateModel(nb, test_data);
  //System.out.println(eval3.toSummaryString());

}

System.out.println(eval3.toSummaryString(folds + "-fold Cross-validation results", false));
System.out.println("Precision: " + eval3.precision(1));
System.out.println("Recall : " + eval3.recall(1));
System.out.println("FMeasure: " + eval3.fMeasure(1));
System.out.println();



System.out.println("Feature Selection: Information Gain ; Classification Algorithm: Random Forest");

int[] arr3 = new int[10];
Evaluation eval4 = new Evaluation(data1);
for (int n = 0; n < folds; n++) {
	
	  Instances train = data1.trainCV(folds, n);
	  Instances test = data1.testCV(folds, n);

	attsel.setEvaluator(infoeval);
	attsel.setSearch(search);
	attsel.SelectAttributes(train);
	//obtain the attribute indices that were selected
	int[] indices = attsel.selectedAttributes();
	for(int i=0; i<9; i++)
	{
		arr3[i] = indices[i];
	}
	arr3[9]= indices[(indices.length)-1];
	//System.out.println(indices.length);
	Remove remove = new Remove();
	remove.setAttributeIndicesArray(arr3);
	//remove.setAttributeIndices(indices);
	remove.setInvertSelection(true);
	remove.setInputFormat(train);
	Instances train_data = Filter.useFilter(train, remove);
	
	remove.setInputFormat(test);
	Instances test_data = Filter.useFilter(test, remove);

	//System.out.println("Attributes selected are: " + Utils.arrayToString(arr3));
	System.out.println("Attributes selected are: " + Utils.arrayToString(indices));

	

RandomForest nb = new RandomForest();
  nb.buildClassifier(train_data);
  eval4.evaluateModel(nb, test_data);
  //System.out.println(eval4.toSummaryString());

}

System.out.println(eval4.toSummaryString(folds + "-fold Cross-validation results", false));
System.out.println("Precision: " + eval4.precision(1));
System.out.println("Recall : " + eval4.recall(1));
System.out.println("FMeasure: " + eval4.fMeasure(1));
System.out.println();







		
	}

}


