using Emgu.CV.Structure;
using Emgu.CV;
using Emgu.CV.Util;

class SourceImage
{
    public SourceImage(string path, Image<Bgr, byte> image, string? subdirectory)
    {
        Path = path;
        Image = image;
        Subdirectory = subdirectory;
    }

    public string Path { get; }
    public Image<Bgr, byte> Image { get; }
    public string? Subdirectory { get; }
    public VectorOfKeyPoint Keypoints { get; internal set; }
    public Mat Descriptors { get; internal set; }
}
