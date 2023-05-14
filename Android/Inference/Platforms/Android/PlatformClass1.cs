using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xamarin.TensorFlow.Lite;

namespace Inference
{
    // All the code in this file is only included on Android.
    public class PlatformClass1
    {
        public void TFLite(string models)
        {
            Stream fileStream = FileSystem.Current.OpenAppPackageFileAsync(models).Result;
            var modelStream = new MemoryStream();
            fileStream.CopyTo(modelStream);
            var _model = modelStream.ToArray();
            var m = Java.Nio.ByteBuffer.Wrap(_model);
            var option = new Interpreter.Options();
            option.SetNumThreads(2);
            var interpreter = new Interpreter(m, option);
            var tensor = interpreter.GetInputTensor(0);
            Console.Write(tensor);


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

    }
}