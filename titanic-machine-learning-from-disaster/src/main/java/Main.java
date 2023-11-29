import com.opencsv.CSVWriter;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
public class Main {
    static final String pathToFolder = "C:\\Users\\stern\\Documents\\EducationLocal\\AI\\rgr\\data\\";

    public static void main(String[] args) throws IOException {
        var titanicSource = getSource(pathToFolder + "prepared_train_data.csv", "Survived");
        var titanicSplitter = new TrainTestSplitter<>(titanicSource, 0.8, 1L);
        var trainingDataset = new MutableDataset<>(titanicSplitter.getTrain());
        var testingDataset = new MutableDataset<>(titanicSplitter.getTest());
        var trainer = new LogisticRegressionTrainer();
        long startTime = System.nanoTime();
        Model<Label> titanicModel = trainer.train(trainingDataset);
        long endTime = System.nanoTime();
        long trainingTime = endTime - startTime;
        var evaluator = new LabelEvaluator();
        startTime = System.nanoTime();
        var evaluation = evaluator.evaluate(titanicModel, testingDataset);
        endTime = System.nanoTime();
        var executionTime = endTime - startTime;
        System.out.println(trainer);
        System.out.println(evaluation.toString());
        System.out.println("Training time in milliseconds: " + (double) trainingTime / 1_000_000);
        System.out.println("Execution time in milliseconds: " + (double) executionTime / 1_000_000);
        titanicSource = getSource(pathToFolder + "prepared_test_data.csv", "");
        List<Prediction<Label>> predictions = titanicModel.predict(new MutableDataset<>(titanicSource));
        saveToFile(predictions);
    }

    private static void saveToFile(List<Prediction<Label>> predictions) throws IOException {
        try (var writer = new CSVWriter(new FileWriter(pathToFolder + "java_result.csv"))) {
            String[] headers = {"PassengerId", "Survived"};
            writer.writeNext(headers, false);
            for (int i = 0; i < predictions.size(); i++) {
                Prediction<Label> p = predictions.get(i);
                String id =Integer.toString(892 + i);
                String value = p.getOutput().getLabel();
                String[] row = {id, value};
                writer.writeNext(row, false);
            }
        }
    }

    private static DataSource<Label> getSource(String pathToSource, String responseName) throws IOException {
        var path = Paths.get(pathToSource);
        var labelFactory = new LabelFactory();
        var csvLoader = new CSVLoader<>(labelFactory);
        return csvLoader.loadDataSource(path, responseName);
    }
}
