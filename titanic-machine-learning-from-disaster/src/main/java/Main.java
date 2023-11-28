import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;

import java.io.IOException;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) throws IOException {
        var path = Paths.get("C:\\Users\\stern\\Documents\\EducationLocal\\AI\\rgr\\data\\prepared_train_data.csv");
        var labelFactory = new LabelFactory();
        var csvLoader = new CSVLoader<>(labelFactory);
        var titanicSource = csvLoader.loadDataSource(path, "Survived");
        var titanicSplitter = new TrainTestSplitter<>(titanicSource, 0.8, 1L);
        var trainingDataset = new MutableDataset<>(titanicSplitter.getTrain());
        var testingDataset = new MutableDataset<>(titanicSplitter.getTest());
        Trainer<Label> trainer = new LogisticRegressionTrainer();
        long startTime = System.nanoTime();
        Model<Label> titanicModel = trainer.train(trainingDataset);
        var evaluator = new LabelEvaluator();
        var evaluation = evaluator.evaluate(titanicModel, testingDataset);
        long endTime = System.nanoTime();
        long executionTime = endTime - startTime;
        System.out.println(trainer);
        System.out.println(evaluation.toString());
        System.out.println("Execution time in nanoseconds: " + executionTime);
        System.out.println("Execution time in milliseconds: " + (double) executionTime / 1_000_000);
    }
}
