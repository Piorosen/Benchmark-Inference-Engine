namespace Android;

public partial class MainPage : ContentPage
{
	int count = 0;

	public MainPage()
	{
		InitializeComponent();
	}

	private void OnCounterClicked(object sender, EventArgs e)
	{
		var data = new Inference.PlatformClass1();

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
        data.TFLite($"tflite/{models[0]}.onnx");

		SemanticScreenReader.Announce(CounterBtn.Text);
        return;

  //      foreach (var model in models)
  //      {
  //          data.Onnx($"Onnx/{model}.onnx");
  //      }
  //      count++;

		//if (count == 1)
		//	CounterBtn.Text = $"Clicked {count} time";
		//else
		//	CounterBtn.Text = $"Clicked {count} times";

	}
}

