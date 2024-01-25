using Emgu.CV.Structure;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Util;

public class ImageRecognizer
{
    public ImageRecognizer(string sourceDirectory)
    {
        SourceDirectory = sourceDirectory;
    }

    private string _lastMessage = "";
    void LogIfChanged(string message)
    {
        if (_lastMessage == message)
            return;

        Console.WriteLine(message);
        _lastMessage = message;
    }

    public void Load()
    {
        _sourceImages = LoadSourceImages(SourceDirectory).ToList();
        ComputeFeaturesForImages(_sourceImages);
    }

    private static IEnumerable<SourceImage> LoadSourceImages(string directoryPath, string? subdirectory = null)
    {
        // Check if the directory exists
        if (!Directory.Exists(directoryPath))
        {
            throw new DirectoryNotFoundException($"The specified directory was not found: {directoryPath}");
        }

        // Get all jpeg files in the directory
        string[] directories = Directory.GetDirectories(directoryPath);

        foreach (var directory in directories)
        {
            var directoryName = Path.GetFileNameWithoutExtension(directory);
            foreach (var image in LoadSourceImages(directory, directoryName))
                yield return image;
        }

        // Get all jpeg files in the directory
        string[] fileEntries = Directory.GetFiles(directoryPath, "*.jpg");
        foreach (var filePath in fileEntries)
        {
            SourceImage? sourceImage = null;
            try
            {
                // Load the image and add to the list
                var image = new Image<Bgr, byte>(filePath);
                sourceImage = new SourceImage(filePath, image, subdirectory);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading image {filePath}: {ex.Message}");
            }

            if (sourceImage is not null)
                yield return sourceImage;
        }
    }

    private static void ComputeFeaturesForImages(IEnumerable<SourceImage> images)
    {
        var detector = new ORB();

        foreach (var sourceImage in images)
        {
            var keypoints = new VectorOfKeyPoint();
            var descriptors = new Mat();
            var image = sourceImage.Image;

            detector.DetectAndCompute(image, null, keypoints, descriptors, false);
            sourceImage.Keypoints = keypoints;
            sourceImage.Descriptors = descriptors;
        }
    }

    public string SourceDirectory { get; }

    private bool _running = false;
    private Thread _runThread;
    private List<SourceImage> _sourceImages;


    public void Start()
    {
        if (_running) throw new InvalidOperationException();

        _running = true;

        _runThread = new Thread(new ThreadStart(Run));
        _runThread.Start();
    }

    public void Stop()
    {
        if (_running is false) throw new InvalidOperationException();

        _running = false;

        _runThread.Join();
    }

    private void Run()
    {
        var detector = new ORB();
        var matcher = new BFMatcher(DistanceType.Hamming);

        var capture = new VideoCapture(0);
        capture.Start();

        while (_running)
        {
            Thread.Sleep(100);

            using var inputFrame = capture.QueryFrame();
            if (inputFrame is null)
                continue;

            using var frame = inputFrame.ToImage<Bgr, byte>();
            if (frame == null)
                continue;

            // Create an empty VectorOfDMatch to store matches
            var matches = new VectorOfDMatch();

            var frameKeypoints = new VectorOfKeyPoint();
            var frameDescriptors = new Mat();
            detector.DetectAndCompute(frame, null, frameKeypoints, frameDescriptors, false);

            double bestScore = 0;

            SourceImage? bestMatch = null;

            for (int i = 0; i < _sourceImages.Count; i++)
            {
                var image = _sourceImages[i];
                var descriptors = image.Descriptors;
                matcher.Match(frameDescriptors, descriptors, matches);

                double score = CalculateScore(matches);
                if (score > bestScore && score > 20)
                {
                    bestScore = score;
                    bestMatch = image;
                }
            }

            if (bestMatch is null)
            {
                //Console.WriteLine("No good match found.");
                continue;
            }

            // Display or use the matched source image
            // sourceImages[bestMatchIndex] is the best match

            var filename = bestMatch.Subdirectory ?? Path.GetFileNameWithoutExtension(bestMatch.Path);

            //LogIfChanged($"Matched image {matchedImage.Path} score: {bestScore}");
            LogIfChanged($"Matched image: {filename}");
        }

        capture.Stop();

    }


    static double CalculateScore(VectorOfDMatch matches, double weight = 0.1)
    {
        double totalDistance = 0;
        int goodMatchCount = 0;

        for (int i = 0; i < matches.Size; i++)
        {
            var match = matches[i];
            if (match.Distance < 30) // Threshold for a "good" match
            {
                totalDistance += match.Distance;
                goodMatchCount++;
            }
        }

        if (goodMatchCount == 0) return 0;

        double averageDistance = totalDistance / goodMatchCount;
        return goodMatchCount - (weight * averageDistance);
    }
}