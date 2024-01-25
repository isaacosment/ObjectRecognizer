using Emgu.CV.Structure;
using Emgu.CV;

class SourceImage
{
    public SourceImage(string path, Image<Bgr, byte> image)
    {
        Path = path;
        Image = image;
    }

    public string Path { get; }
    public Image<Bgr, byte> Image { get; }
}
