// Example: Convert text to lowercase
String text = "This is an Example Text";
text = text.toLowerCase();
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
// Load and prepare the dataset
ArffLoader loader = new ArffLoader();
loader.setFile(new File("consumer_complaints.arff"));
Instances dataset = loader.getDataSet();
dataset.setClassIndex(dataset.numAttributes() - 1);
// Create and configure the classifier (Support Vector Machine)
Classifier classifier = new SMO();
// Train the classifier
classifier.buildClassifier(dataset);
import weka.classifiers.Evaluation;
// Load testing data
ArffLoader testLoader = new ArffLoader();
testLoader.setFile(new File("test_data.arff"));
Instances testDataset = testLoader.getDataSet();
testDataset.setClassIndex(testDataset.numAttributes() - 1);
// Evaluate the model
Evaluation evaluation = new Evaluation(dataset);
evaluation.evaluateModel(classifier, testDataset);
System.out.println(evaluation.toSummaryString());
// Load new data for prediction
ArffLoader newLoader = new ArffLoader();
newLoader.setFile(new File("new_data.arff"));
Instances newData = newLoader.getDataSet();
newData.setClassIndex(newData.numAttributes() - 1);
// Make predictions on new data
for (int i = 0; i < newData.numInstances(); i++) {
    double prediction = classifier.classifyInstance(newData.instance(i));
    System.out.println("Prediction for instance " + (i + 1) + ": " + dataset.classAttribute().value((int) prediction));
}
