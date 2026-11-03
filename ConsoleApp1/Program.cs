using System.Text.RegularExpressions;
using OpenCvSharp;

class Program
{
    public static void Main(string[] args)
    {
        string providedPath = "";
        bool captureOpened = false;
        while (!captureOpened)
        {
            Console.Write("Input a video's file path:");
            providedPath = Console.ReadLine()?.Trim() ?? "";
            if (!File.Exists(providedPath) && ( providedPath.EndsWith(".MOV") || providedPath.EndsWith(".MP4")))
            {
                Console.WriteLine("Could not find video at path\n");
            }
            else
            {
                captureOpened = true;
            }
        }

        bool running = true;
        int threshold = 165;
        
        Task videoFeed = Task.Run(() =>
        {
            VideoCapture capture = new VideoCapture(providedPath);
            //Console.WriteLine("Opened: " + capture.IsOpened());
            Mat currentFrame = new Mat();
            Cv2.NamedWindow("mat frame", WindowFlags.KeepRatio);
            Cv2.NamedWindow("current frame", WindowFlags.KeepRatio);

            while (capture.Read(currentFrame))
            {
                if (!running) continue;
                
                using var mat = currentFrame.Clone();
                Cv2.GaussianBlur(mat, mat, new Size(5, 5), 1);
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2GRAY);
                Cv2.Threshold(mat, mat, threshold, 255, ThresholdTypes.Binary);
                Cv2.FindContours(mat, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External,
                    ContourApproximationModes.ApproxSimple);

                Cv2.DrawContours(currentFrame, contours, -1, new Scalar(0, 255, 0), 2);
                Cv2.ImShow("mat frame", mat);
                Cv2.ImShow("current frame", currentFrame);
                int waitKey = Cv2.WaitKey(1);
                if (waitKey == 27)
                {
                    running = false;
                    break;
                };
            }

            Cv2.DestroyAllWindows();
        });
            
        
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
                        videoFeed.Dispose();
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
    }
}