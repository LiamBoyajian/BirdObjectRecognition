using OpenCvSharp;

class Program
{
    public static void Main(string[] args)
    {
        VideoCapture capture = null;
        bool captureOpened = false;
        while (!captureOpened)
        {
            Console.Write("Input a video's file path:");
            String ProvidedPath = Console.ReadLine()?.Trim() ?? "";
            capture = new VideoCapture(ProvidedPath);
            if (!capture.IsOpened())
            {
                Console.WriteLine("Could not open video at path\n");
            }
            else
            {
                captureOpened = true;
            }
        }
        
        Console.WriteLine("Opened: " + capture?.IsOpened());
        //loop through video frames
        Mat mat = new Mat();
        while (capture?.Read(mat) ?? false)
        {
            Console.WriteLine(capture.Get(VideoCaptureProperties.PosFrames));
            Cv2.ImShow("display: " + mat, mat);
            if (Cv2.WaitKey(30) == 27) break;
        }
        Cv2.DestroyAllWindows();
        mat.Dispose();
    }
}