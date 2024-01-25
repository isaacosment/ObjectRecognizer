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
        _sourceImages = LoadSourceImages(SourceDirectory);

        var images = _sourceImages.Select(s => s.Image).ToList();
        _sourceFeatures = ComputeFeaturesForImages(images);
    }

    private static List<SourceImage> LoadSourceImages(string directoryPath)
    {
        var images = new List<SourceImage>();

        // Check if the directory exists
        if (!Directory.Exists(directoryPath))
        {
            throw new DirectoryNotFoundException($"The specified directory was not found: {directoryPath}");
        }

        // Get all jpeg files in the directory
        string[] fileEntries = Directory.GetFiles(directoryPath, "*.jpg");
        foreach (var filePath in fileEntries)
        {
            try
            {
                // Load the image and add to the list
                var image = new Image<Bgr, byte>(filePath);
                images.Add(new(filePath, image));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading image {filePath}: {ex.Message}");
            }
        }

        return images;
    }

    private static List<(VectorOfKeyPoint, Mat)> ComputeFeaturesForImages(List<Image<Bgr, byte>> images)
    {
        var featuresList = new List<(VectorOfKeyPoint, Mat)>();
        var detector = new ORB();

        foreach (var image in images)
        {
            var keypoints = new VectorOfKeyPoint();
            var descriptors = new Mat();
            detector.DetectAndCompute(image, null, keypoints, descriptors, false);
            featuresList.Add((keypoints, descriptors));
        }

        return featuresList;
    }

    public string SourceDirectory { get; }

    private bool _running = false;
    private Thread _runThread;
    private Task _runTask;
    private List<SourceImage> _sourceImages;
    private List<(VectorOfKeyPoint, Mat)> _sourceFeatures;


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
            int bestMatchIndex = -1;

            for (int i = 0; i < _sourceFeatures.Count; i++)
            {
                var feature = _sourceFeatures[i];
                matcher.Match(frameDescriptors, feature.Item2, matches);

                double score = CalculateScore(matches);
                if (score > bestScore && score > 20)
                {
                    bestScore = score;
                    bestMatchIndex = i;
                }
            }

            if (bestMatchIndex == -1)
            {
                //Console.WriteLine("No good match found.");
                continue;
            }

            // Display or use the matched source image
            // sourceImages[bestMatchIndex] is the best match

            var matchedImage = _sourceImages[bestMatchIndex];

            var filename = Path.GetFileNameWithoutExtension(matchedImage.Path);

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