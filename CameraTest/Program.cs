string imageDirectory = @"C:\files\image-lookup\source-images";

Console.WriteLine($"Application starting. Source image directory is {imageDirectory}");


var recognizer = new ImageRecognizer(imageDirectory);

recognizer.Load();

recognizer.Start();

bool _quit = false;

Console.CancelKeyPress += (sender, e) =>
{
    _quit = true;
    e.Cancel = true;
};

Console.WriteLine("Recognizer running. Press Ctrl+C to end.");

while (!_quit)
{
    Thread.Sleep(250);
}

Console.WriteLine("Stopping recognizer.");

recognizer.Stop();

Console.WriteLine("Recognizer stopped.");