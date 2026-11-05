using System.Text.RegularExpressions;
using OpenCvSharp;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.Models;
using SkiaSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    public static void Main(string[] args)
    {
        //LinkedList<string> lastCommands = new LinkedList<string>();
        
        string providedPath = "";
        bool validFile = false;
        while (!validFile)
        {
            Console.Write("Input a video's file path:");
            providedPath = Console.ReadLine()?.Trim() ?? "";
            if (!File.Exists(providedPath) && ( providedPath.EndsWith(".MOV") || providedPath.EndsWith(".MP4")))
            {
                Console.WriteLine("Could not find video at path\n");
            }
            else
            {
                Console.WriteLine("Valid path input.\n---------------------- \n");
                validFile = true;
            }
        }

        bool running = true;
        int threshold = 165;
        
        Task terminal = Task.Run(() =>
        {
            bool inputToTerminal = true;
            while (inputToTerminal)
            {
                Console.Write(">");
                string? input = Console.ReadLine();
                if (input == null) continue;
            
                string[] arg =  input.Split();
                if (arg.Length > 0)
                {
                    switch (arg[0])
                    {
                        case "pause":
                            running = false;
                            break;
                        case "play":
                            running = true;
                            break;
                        case ("exit"):
                            inputToTerminal = false;
                            break;
                        case ("threshold"): 
                            threshold = int.Parse(arg[1]); 
                            Console.WriteLine("THRESHOLD UPDATED: " + threshold);
                            break;
                        case ("help"):
                            void PrintHelpLine(string commandName, string explaination)
                            {
                                Console.WriteLine(commandName.ToUpper());
                                Console.WriteLine("     " + explaination);
                            }
                            PrintHelpLine("play", "Resume video stream");
                            PrintHelpLine("pause", "Pause video stream");
                            PrintHelpLine("exit", "Exit program");
                            PrintHelpLine("threshold", "Change threshold value for threshold function");
                            break;
                        default:
                            break;
                    }
                }
            
            }
        });
            Yolo yolo = null;
            try
            {
                yolo = new Yolo(new YoloOptions
                {
                    ExecutionProvider = new CudaExecutionProvider(Path.Combine(AppContext.BaseDirectory, "Data", "onnx", "yolov8n-cls.onnx")),
                });
            }
            catch (Exception e)
            {
                Console.WriteLine("Error: " + e.Message);
                return;
            }
            if (yolo == null) return;

            
            VideoCapture capture = new VideoCapture(providedPath);
            Console.WriteLine("Opened: " + capture.IsOpened());
            Mat currentFrame = new Mat();
            //Cv2.NamedWindow("mat frame", WindowFlags.KeepRatio);
            Cv2.NamedWindow("current frame", WindowFlags.KeepRatio);

            Console.WriteLine("Is open" + capture.IsOpened());
            capture.Read(currentFrame);
            while (capture.IsOpened())
            {
                
                if (!running)
                {
                    Thread.Sleep(50);
                    continue;
                }
                if (!capture.Read(currentFrame) || currentFrame.Empty()) break;

                // ----- IMAGE ENCODING -----
                
                //using var mat = currentFrame.Clone();
                using var currentFrameAnnotated = currentFrame.Clone();
                
                //Cv2.GaussianBlur(mat, mat, new Size(5, 5), 1);
                //Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2GRAY);
                //Cv2.Threshold(mat, mat, threshold, 255, ThresholdTypes.Binary);
                //Cv2.FindContours(mat, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External,
                //    ContourApproximationModes.ApproxSimple);

                //Cv2.DrawContours(currentFrame, contours, -1, new Scalar(0, 255, 0), 2);

                // ----- YOLO PIPELINE -----
                
                var encodedData = currentFrame.ToBytes(".jpg");
                using var skImagae = SKImage.FromEncodedData(encodedData);

                var results = yolo.RunClassification(skImagae);

                // ----- DISPLAY RESULTS -----
                
                //temp ai code:
                int yOffset = 40;
                foreach (var result in results.Take(3)) // Display the top 3 predictions
                {
                    string text = $"{result.Label}: {result.Confidence:P1}";
                    Cv2.PutText(
                        currentFrameAnnotated, 
                        text, 
                        new Point(20, yOffset), 
                        HersheyFonts.HersheySimplex, 
                        0.8, 
                        new Scalar(0, 255, 0), 
                        2
                    );
                    yOffset += 30;
                }
                
                //Cv2.ImShow("mat frame", mat);
                Cv2.ImShow("current frame", currentFrameAnnotated);
                
                // ----- OPTIONS -----
                int waitKey = Cv2.WaitKey(30);
                if (waitKey == 27)
                {
                    running = false;
                    break;
                };
            }

            Cv2.DestroyAllWindows();
            capture.Dispose();
            yolo.Dispose();
        
        // ----- TERMINAL -----

        //LinkedListNode<string>? terminalCommand = null;
        
    }
}