namespace Loralei.Tests;

/// <summary>
/// Serialises every test that spawns loralei, across both test assemblies.
/// <para>
/// `dotnet test` runs the assemblies concurrently, but there is one GPU and one workspace
/// `loralei.yaml` behind all of these tests. Two of them at once means fighting over VRAM and
/// over that file, which fails in ways that look like real bugs. A lock file is the only
/// thing that reaches across processes.
/// </para>
/// </summary>
internal sealed class CliLock : IDisposable
{
    private static readonly string Path =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), "loralei-tests.lock");

    private readonly FileStream _stream;

    private CliLock(FileStream stream) => _stream = stream;

    public static CliLock Acquire(TimeSpan? timeout = null)
    {
        var deadline = DateTime.UtcNow + (timeout ?? TimeSpan.FromMinutes(30));
        while (true)
        {
            try
            {
                return new CliLock(new FileStream(Path, FileMode.OpenOrCreate,
                    FileAccess.ReadWrite, FileShare.None));
            }
            catch (IOException) when (DateTime.UtcNow < deadline)
            {
                Thread.Sleep(250);
            }
        }
    }

    public void Dispose() => _stream.Dispose();
}
