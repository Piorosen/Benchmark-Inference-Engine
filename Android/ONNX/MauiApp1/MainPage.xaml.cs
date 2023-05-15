using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MauiApp1;

public partial class MainPage : ContentPage
{
	int count = 0;

	public MainPage()
	{
		InitializeComponent();
	}

    public void TFLite(string models)
    {

        Stream fileStream = FileSystem.Current.OpenAppPackageFileAsync(models).Result;
        var modelStream = new MemoryStream();
        fileStream.CopyTo(modelStream);
        var _model = modelStream.ToArray();


        var bb = Java.Nio.MappedByteBuffer.Wrap(_model);
        

        var ip= new Xamarin.TensorFlow.Lite.Interpreter(bb);
        var shape = ip.GetInputTensor(0).Shape();
        Console.WriteLine(string.Join(", ", shape));

        //np_features = np.random.rand(input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]).astype(input_details[0]['dtype'])


        //ip.Run()
    }

    public void Onnx(string models)
    {
        SessionOptions sess_options = new SessionOptions();
        //sess_options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        //sess_options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        //sess_options.AppendExecutionProvider("XNNPACK");
        sess_options.AppendExecutionProvider_CPU(1);
        sess_options.AppendExecutionProvider_Nnapi(NnapiFlags.NNAPI_FLAG_CPU_ONLY);

        Stream fileStream = FileSystem.Current.OpenAppPackageFileAsync(models).Result;
        var modelStream = new MemoryStream();
        fileStream.CopyTo(modelStream);
        var _model = modelStream.ToArray();

        InferenceSession session = new InferenceSession(_model, sess_options);
        var model_input = session.InputMetadata.First().Value.Dimensions;
        var model_shape = model_input.Aggregate((num1, num2) => num1 * num2);
        var model_input_name = session.InputMetadata.First().Key;
        Console.WriteLine($"{model_input_name} 모델 이름");
        Console.WriteLine($"{string.Join(", ", model_input)} dim");

        Console.WriteLine("setup");

        Random r = new Random();

        for (int e = 0; e < 100; e++)
        {
            float[] random_shape = new float[model_shape];
            for (int j = 0; j < model_shape; j++)
            {
                random_shape[j] = (float)r.NextDouble();
            }

            var inputTensor = new DenseTensor<float>(random_shape, model_input);
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor($"{model_input_name}", inputTensor) };

            long start = DateTime.UtcNow.Ticks;
            var result = session.Run(input);
            long end = DateTime.UtcNow.Ticks;
            Console.WriteLine($"{models} inference {new TimeSpan(end - start).TotalMilliseconds} ms");
        }
    }

    private void OnCounterClicked(object sender, EventArgs e)
	{
        var models = new string[]
        {
            "alexnet",
            "mobilenet_v2",
            "resnet18",
            "resnet50",
            "resnet101",
            "vgg16",
            "googlenet",
            //"alexnet-qint8",
            //"googlenet-qint8",
            //"mobilenet_v2-qint8",
            //"resnet18-qint8",
            //"resnet50-qint8",
            //"resnet101-qint8",
            //"vgg16-qint8",
        };

        foreach (var model in models)
        {
            TFLite($"tflite/{model}.tflite");
        }

        count++;

		if (count == 1)
			CounterBtn.Text = $"Clicked {count} time";
		else
			CounterBtn.Text = $"Clicked {count} times";

		SemanticScreenReader.Announce(CounterBtn.Text);
	}
}

