using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

public class Program
{
    public class ModelInput
    {
        [VectorType(4)]
        [ColumnName("dense_4_input")]
        public float[]? Input { get; set; }
    }

    public class ModelOutput
    {
        [VectorType(3)]
        [ColumnName("dense_5")]
        public float[]? Output { get; set; }
    }

    public static void Main(string[] args)
    {
        var context = new MLContext();

        var data = new[] { new ModelInput { Input = new float[] { 0.1f, 0.2f, 0.3f, 0.4f } } };
        var dataView = context.Data.LoadFromEnumerable(data);

        string modelPath = "model_01.onnx";

        var pipeline = context.Transforms.ApplyOnnxModel(outputColumnNames: new[] { "dense_5" }, inputColumnNames: new[] { "dense_4_input" }, modelFile: modelPath);

        var transformedValues = pipeline.Fit(dataView).Transform(dataView);

        var output = context.Data.CreateEnumerable<ModelOutput>(transformedValues, reuseRowObject: false);

        foreach (var prediction in output)
        {
            Console.WriteLine($"Prediction: {string.Join(", ", prediction.Output)}");
        }
    }
}
